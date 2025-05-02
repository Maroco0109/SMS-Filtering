import torch
import argparse
import pytorch_lightning as pl
import torch.nn.functional as F

from utils import load_model, Logger

from dataloader import PlmData
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from torchmetrics import Accuracy, F1Score
from transformers.modeling_outputs import SequenceClassifierOutput

import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from utils.model_util import load_model

logger = Logger('model-log', 'log/')

class FocalLoss(torch.nn.Module):
    '''
    Focal Loss 함수 구현
    '''    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

'''
Description
-----------
Hate-Speech Detection with Transformer Models

Models
------
huggingface에 공개된 한국어 사전학습 모델 사용

    BERT: monologg/kobert
    ELECTRA: monologg/koelectra-base-v3-discriminator
    BigBird: monologg/kobigbird-bert-base (현재 비공개 처리됨)
    RoBERTa: klue/roberta-base     
'''
# 추가 레이어 생성
class CustomClassifier(torch.nn.Module):
    def __init__(self, hidden_size, num_labels=2):
        super(CustomClassifier, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_labels)
        )

    def forward(self, x):
        return self.classifier(x)
    
class LightningPLM(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.validation_step_outputs = []
        
        self.model_type = self.hparams.model_type.lower()
        self.model_name = self.hparams.model_name
        self.num_labels = self.hparams.num_labels
        self.use_focal_loss = getattr(self.hparams, "use_focal_loss", False)
        self.use_custom_classifier = getattr(self.hparams, "use_custom_classifier", False)
        self.threshold = getattr(self.hparams, "threshold", 0.5)
        self.max_len = getattr(self.hparams, "max_len", 128)
        self.lr = getattr(self.hparams, "lr", 2e-5)
        self.warmup_ratio = getattr(self.hparams, "warmup_ratio", 0.1)

        self.model, self.tokenizer = load_model(self.model_type, self.num_labels)

        # Freeze first 4 encoder layers
        if getattr(self.hparams, "freeze_encoder", False):
            try:
                encoder = getattr(self.model, self.model.base_model_prefix).encoder
                for i in range(4):
                    for param in encoder.layer[i].parameters():
                        param.requires_grad = False
                print("✅ Encoder layer 0 ~ 3 freeze 완료")
            except AttributeError:
                print("⚠️ Encoder layer freeze 실패 (base_model_prefix 없음)")



        hidden_size = self.model.config.hidden_size
        if self.use_custom_classifier:
            self.custom_classifier = CustomClassifier(hidden_size, num_labels=self.num_labels)
            print("✅ Using custom classifier layers")
        else:
            print("✅ Using default classifier head from pretrained model")

        if self.use_focal_loss:
            if FocalLoss is None:
                raise ImportError("FocalLoss module is not available. Please ensure loss/focal_loss.py exists.")
            self.loss_function = FocalLoss(alpha=1.0, gamma=2.0)
            print("✅ Using Focal Loss")
        else:
            self.loss_function = nn.CrossEntropyLoss()
            print("✅ Using CrossEntropyLoss")

        self.softmax = nn.Softmax(dim=1)
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
        parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for scheduler')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
        parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
        parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss if set')
        parser.add_argument('--use_custom_classifier', action='store_true', help='Use custom classifier if set')
        parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder layers 0~3')
        return parser


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )

        pooled_output = output.hidden_states[-1][:, 0, :]

        if self.use_custom_classifier:
            logits = self.custom_classifier(pooled_output)
        else:
            logits = output.logits

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions
        )
        
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        acc = self.accuracy(preds, labels)
        f1 = self.f1_score(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        probs = self.softmax(outputs.logits)
        preds = (probs[:, 1] >= self.threshold).long()
        acc = self.accuracy(preds, labels)
        f1 = self.f1_score(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.validation_step_outputs.append({
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1
        })
        return {"val_loss": loss, "val_acc": acc, "val_f1": f1}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        probs = self.softmax(outputs.logits)
        preds = (probs[:, 1] >= self.threshold).long()
        acc = self.accuracy(preds, labels)
        f1 = self.f1_score(preds, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc, "test_f1": f1}

    def on_validation_epoch_end(self):
        avg_losses = []
        avg_accuracies = []
        avg_f1s = []

        for output in self.validation_step_outputs:
            avg_losses.append(output['val_loss'])
            avg_accuracies.append(output['val_acc'])
            avg_f1s.append(output['val_f1'])

        self.log_dict({
            'avg_val_loss': torch.stack(avg_losses).mean(),
            'avg_val_acc': torch.stack(avg_accuracies).mean(),
            'avg_val_f1': torch.stack(avg_f1s).mean()
        })

        # 메모리에서 리스트 초기화 (중복 방지)
        self.validation_step_outputs.clear()

    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.05,   # 25.05.02 0.01 -> 0.05
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        total_steps = len(self.train_dataloader()) * self.trainer.max_epochs
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data_path = f'{self.hparams.data_dir}/train.csv'
        self.train_set = PlmData(data_path, tokenizer=self.tokenizer, \
            max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader
    
    def val_dataloader(self):
        data_path = f'{self.hparams.data_dir}/valid.csv'
        self.valid_set = PlmData(data_path, tokenizer=self.tokenizer, \
            max_len=self.hparams.max_len)
        val_dataloader = DataLoader(
            self.valid_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=False, collate_fn=self._collate_fn)
        return val_dataloader
