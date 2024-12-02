import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import json
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertTokenizer, BertModel, get_scheduler
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm.notebook import tqdm
from kobert_tokenizer import KoBERTTokenizer
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from scripts.KoBERTDataset import KoBERTDataset
from scripts.model import BERTClassifier
from scripts.utils import FocalLoss, calc_accuracy


def prepare_data_loaders():
    # GPU 사용 설정
    device = torch.device("cuda:0")

    # 디렉토리 읽기
    with open("/json/config.json","r") as f:
        config = json.load(f)

    # 파일 로드
    train_df = pd.read_csv(config(["train"]["input_dir"]))
    test_df = pd.read_csv(config(["test"]["input_dir"]))

    train_texts = train_df['text'].tolist()  # 텍스트 열에서 리스트로 변환
    train_labels = train_df['label'].tolist()  # 레이블 열에서 리스트로 변환

    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    bert_model = BertModel.from_pretrained("monologg/kobert")
    model = BERTClassifier(bert_model).to(device)

    # 레이블 인코딩 ('ham'과 'spam'을 각각 0과 1로 변환)
    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['label'])  # 'ham' → 0, 'spam' → 1
    test_df['label'] = label_encoder.transform(test_df['label'])

    # 데이터셋 및 DataLoader 준비
    train_dataset = KoBERTDataset(train_texts, train_labels, tokenizer, max_len=128)
    test_dataset = KoBERTDataset(test_texts, test_labels, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    return train_loader, test_loader

def main():
    ##GPU 사용 시
    device = torch.device("cuda:0")

    # 설정 파일 읽기
    with open("/json/config.json","r") as f:
        config = json.load(f)

    # 파일 로드
    train_df = pd.read_csv(config(["train"]["input_dir"]))
    test_df = pd.read_csv(config(["test"]["input_dir"]))

    # Hyperparameters
    max_len = 128
    batch_size = 64
    learning_rate = 1e-5
    num_epochs = 3

    prepare_data_loaders()

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    bert_model = BertModel.from_pretrained("monologg/kobert")
    model = BERTClassifier(bert_model).to(device)

    # 레이블 인코딩 ('ham'과 'spam'을 각각 0과 1로 변환)
    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['label'])  # 'ham' → 0, 'spam' → 1
    test_df['label'] = label_encoder.transform(test_df['label'])

    # Dataset 생성
    train_texts = train_df['text'].tolist()  # 텍스트 열에서 리스트로 변환
    train_labels = train_df['label'].tolist()  # 레이블 열에서 리스트로 변환
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    train_dataset = KoBERTDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    test_dataset = KoBERTDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_loader, test_loader = prepare_data_loaders()
    
    # 클래스 가중치 계산
    classes = np.unique(train_labels)  # [0, 1]을 numpy 배열로 변환
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 손실 함수 정의
    loss_fn = FocalLoss(alpha=2.0, gamma=2.0)

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 스케쥴러 설정
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs  # 전체 학습 스텝 수
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 훈련 및 검증
    for epoch in range(num_epochs):
        model.train()
        total_acc, total_loss = 0, 0

        # Training phase
        for batch in train_loader:
            optimizer.zero_grad()

            # Extract inputs and labels
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_acc += calc_accuracy(outputs, labels)

        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f} | Train Accuracy: {total_acc / len(train_dataset):.4f}")

        # Validation phase
        model.eval()
        total_acc, total_loss = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                # Extract inputs and labels
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                token_type_ids = batch[2].to(device)
                labels = batch[3].to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(outputs, labels)

                total_loss += loss.item()
                total_acc += calc_accuracy(outputs, labels)

        print(f"Epoch {epoch+1} | Val Loss: {total_loss / len(test_loader):.4f} | Val Accuracy: {total_acc / len(test_dataset):.4f}")

    # Debugging logits
    for batch in test_loader:
        texts, labels = batch[0], batch[3]

        # Ensure texts is a list of strings
        if isinstance(texts, torch.Tensor):
            texts = [str(t) for t in texts]

        # Tokenizer conversion
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        # Move inputs to GPU
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels.to(device)

        # Model prediction
        outputs = model(**inputs)
        print("Logits:", outputs)

    # 모델 저장
    output_dir = config(["model"]["output_file"])
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved in {output_dir}")




# 메인 함수 실행
if __name__ == '__main__':
    main()