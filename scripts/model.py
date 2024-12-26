import torch
from transformers import BertTokenizer, BertModel
from torch import nn

##GPU 사용 시
device = torch.device("cuda:0")
# 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.bert = bert  # Pre-trained BERT 모델
        self.classifier = nn.Sequential(
            nn.Linear(bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 이진 분류
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output  # [CLS] 토큰의 임베딩
        return self.classifier(pooled_output)

class tokenizers_models():
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    bert_model = BertModel.from_pretrained("monologg/kobert")
    model = BERTClassifier(bert_model).to(device)