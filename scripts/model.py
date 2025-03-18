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
        self.bert = bert
        
        # Freeze some of the BERT layers to prevent overfitting
        for param in list(bert.parameters())[:-2]:  # Freeze all except last 2 layers
            param.requires_grad = False
            
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        self.classifier = nn.Sequential(
            nn.Linear(bert.config.hidden_size, 64),
            nn.LayerNorm(64),  # Add layer normalization
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
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