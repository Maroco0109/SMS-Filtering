import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import torch.nn.functional as F

# KoBERT 모델 정의
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

# 감성 추론 함수
def predict_sentiment(text, model, tokenizer, max_len=128):
    """
    주어진 텍스트에 대해 감성 분석 (spam 또는 ham) 추론을 수행합니다.

    Args:
        text (str): 입력 텍스트.
        model (nn.Module): 학습된 KoBERT 모델.
        tokenizer (BertTokenizer): KoBERT 토크나이저.
        max_len (int): 최대 토큰 길이.

    Returns:
        tuple: (예측 클래스, spam 확률, ham 확률)
    """
    model.eval()  # 평가 모드
    with torch.no_grad():
        # 입력 텍스트 토큰화
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

        # 모든 텐서를 동일한 디바이스로 이동
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # `token_type_ids`가 없는 경우 기본값 생성 후 디바이스 이동
        if 'token_type_ids' in inputs:
            token_type_ids = inputs['token_type_ids'].to(device)
        else:
            token_type_ids = torch.zeros_like(input_ids).to(device)

        # 모델 추론
        logits = model(input_ids, attention_mask, token_type_ids)

        # 확률 계산
        probs = F.softmax(logits, dim=1)  # 소프트맥스를 사용해 확률 계산
        spam_prob, ham_prob = probs[0][1].item(), probs[0][0].item()  # spam: 1, ham: 0

        # 예측 클래스
        predicted_class = "spam" if spam_prob > ham_prob else "ham"

        return predicted_class, spam_prob, ham_prob

# 메인 함수
if __name__ == "__main__":
    # GPU/CPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 및 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    bert_model = BertModel.from_pretrained("monologg/kobert")
    model = BERTClassifier(bert_model).to(device)

    # 학습된 모델 가중치 로드
    model_path = "/home/maroco/dataset/model/sentiment_model.pt"  # 모델 가중치 파일 경로 수정
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 테스트 입력
    while True:
        text = input("입력 텍스트 (종료하려면 'exit' 입력): ")
        if text.lower() == "exit":
            print("프로그램 종료.")
            break
        predicted_class, spam_prob, ham_prob = predict_sentiment(text, model, tokenizer)
        print(f"입력 텍스트: {text}")
        print(f"예측 결과: {predicted_class}")
        print(f"spam 확률: {spam_prob:.2%}, ham 확률: {ham_prob:.2%}")