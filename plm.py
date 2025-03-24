import torch
import argparse
import pytorch_lightning as pl
import torch.nn.functional as F

from utils import load_model, Logger

from dataloader import PlmData
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AdamW, get_linear_schedule_with_warmup
from torchmetrics import Accuracy
from transformers.modeling_outputs import SequenceClassifierOutput


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
class LightningPLM(LightningModule):
    def __init__(self, hparams):
        super(LightningPLM, self).__init__()
        self.save_hyperparameters(hparams)
        self.validation_step_outputs = []
        self.accuracy = Accuracy(task="binary")
        self.softmax = torch.nn.Softmax(dim=-1)

        self.model_type = hparams.model_type.lower()
        self.model, self.tokenizer = load_model(model_type=self.model_type, num_labels=self.hparams.num_labels)
        # 손실 함수 설정
        if getattr(hparams, 'use_focal_loss', False):
            self.loss_function = FocalLoss(alpha=1.0, gamma=2.0)
            print("✅ Using Focal Loss")
        else:
            self.loss_function = torch.nn.CrossEntropyLoss()
            print("✅ Using CrossEntropyLoss")
        
        self.dropout = torch.nn.Dropout(0.3)      # ✅ Dropout 추가
        
        # freeze
        self.freeze_encoder_layers(num_layers=4)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch-size',
                            type=int,
                            default=32,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=1e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.2,    # warmup ratio 수정
                            help='warmup ratio')
        return parser

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.int).type_as(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, \
            token_type_ids=token_type_ids, labels=labels, return_dict=True)
        
        logits = self.dropout(output.logits)    # ✅ Dropout 적용
        return SequenceClassifierOutput(  # ✅ 직접 생성하여 반환
            loss=output.loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions
        )

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        preds = self.softmax(output.logits).argmax(dim=-1)  # ✅ `argmax()` 추가
        self.log_dict({
            'train_loss' : output.loss,
            'train_acc' : self.accuracy(preds, label)
        }, prog_bar=True)

        return output.loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        acc = self.accuracy(self.softmax(output.logits).argmax(dim=-1), label)

        # 결과 저장
        self.validation_step_outputs.append({'val_loss': output.loss, 'val_acc': acc})

        self.log_dict({
            'val_loss': output.loss,
            'val_acc': acc
        }, prog_bar=True, on_step=False, on_epoch=True)

        return output.loss


    def on_validation_epoch_end(self):
        avg_losses = []
        avg_accuracies = []

        for output in self.validation_step_outputs:
            avg_losses.append(output['val_loss'])
            avg_accuracies.append(output['val_acc'])

        self.log_dict({
            'avg_val_loss': torch.stack(avg_losses).mean(),
            'avg_val_acc': torch.stack(avg_accuracies).mean()
        })

        # 메모리에서 리스트 초기화 (중복 방지)
        self.validation_step_outputs.clear()

    
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer \
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer \
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        train_total = len(self.train_dataloader()) * self.trainer.max_epochs
        warmup_steps = int(train_total * self.hparams.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(    # Cosine Decay 적용
            optimizer,
            num_warmup_steps=warmup_steps, 
            num_training_steps=train_total)
        lr_scheduler = {'scheduler': scheduler, 'name': 'get_linear_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

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
    
    def freeze_encoder_layers(self, num_layers=4):
        """
        Transformer 모델의 초기 num_layers 개 레이어를 freeze하여 과적합 방지
        """
        if hasattr(self.model, "bert"):
            encoder = self.model.bert.encoder
        elif hasattr(self.model, "electra"):
            encoder = self.model.electra.encoder
        elif hasattr(self.model, "roberta"):
            encoder = self.model.roberta.encoder
        else:
            print("❌ Freeze 대상 인코더를 찾을 수 없습니다.")
            return

        for layer_idx in range(num_layers):
            for param in encoder.layer[layer_idx].parameters():
                param.requires_grad = False

        print(f"✅ Encoder layer 0 ~ {num_layers - 1} freeze 완료")