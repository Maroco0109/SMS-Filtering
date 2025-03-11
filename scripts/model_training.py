import torch
import pandas as pd
import numpy as np
import json
import os
import glob
from transformers import AdamW, get_scheduler
from sklearn.utils.class_weight import compute_class_weight
from model import tokenizers_models
from utils import FocalLoss, calc_accuracy
from config import Config
from data_loader import prepare_data_loaders

def main():
    ##GPU 사용 시
    device = torch.device("cuda:0")
    
    # 현재 스크립트 위치 기준으로 config.json 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "json/config.json")

     # 설정 파일 읽기
    with open(config_path,"r",encoding="utf-8") as f:
        config = json.load(f)

    # 파일 로드
    train_files = glob.glob(config["train"]["input_dir"])
    test_files = glob.glob(config["test"]["input_dir"])
    
    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

    # 데이터 로더 준비
    train_loader, test_loader, train_dataset, test_dataset, train_labels, test_labels = prepare_data_loaders()
    
    # 클래스 가중치 계산
    classes = np.unique(train_labels)  # [0, 1]을 numpy 배열로 변환
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # KoBERT 모델과 토크나이저 로드
    tokenizer_model = tokenizers_models()
    model = tokenizer_model.model
    tokenizer = tokenizer_model.tokenizer

    # 손실 함수 정의
    loss_fn = FocalLoss(alpha=2.0, gamma=2.0)

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # 스케쥴러 설정
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * Config.NUM_EPOCHS  # 전체 학습 스텝 수
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 훈련 및 검증
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_acc, total_loss = 0, 0
        for batch in train_loader:
            input_ids, attention_mask, token_type_ids, labels = batch
            print(f"input_ids.shape: {input_ids.shape}")
            print(f"attention_mask.shape: {attention_mask.shape}")
            print(f"token_type_ids.shape: {token_type_ids.shape}")
            print(f"labels.shape: {labels.shape}")
            break
        
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
    output_dir = config["model"]["output_file"]
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved in {output_dir}")



# 메인 함수 실행
if __name__ == '__main__':
    main()