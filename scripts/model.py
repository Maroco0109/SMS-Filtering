import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from transformers import AutoTokenizer
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
        # TorchScript compatibility: explicitly specify return_dict=False
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        pooled_output = outputs[1]  # [CLS] 토큰의 임베딩
        return self.classifier(pooled_output)

    @torch.jit.export
    def get_embedding_dim(self):
        """Export this method for TorchScript"""
        return self.bert.config.hidden_size

class tokenizers_models:
    def __init__(self):
        print("KoBERT 모델 및 토크나이저 로드 중...")
        
        # KoBERT 토크나이저 로드
        self.tokenizer_path = get_tokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
        
        # KoBERT 모델 로드
        self.bert_model, self.vocab = get_pytorch_kobert_model()
        
        # BERT Classifier 초기화 
        self.model = BERTClassifier(self.bert_model).to(device)
        
        print("KoBERT 모델 및 토크나이저 로드 완료!")
        
# 클래스 인스턴스 생성 (전역으로 사용 가능)
tokenizer_model = tokenizers_models()