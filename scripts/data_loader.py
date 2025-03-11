import torch
import json
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from model import tokenizers_models
from KoBERTDataset import KoBERTDataset
from config import Config

# Data loaders
def prepare_data_loaders():
    # GPU 사용 설정
    device = torch.device("cuda:0")

    # 현재 스크립트 위치 기준으로 config.json 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "json/config.json")

     # 설정 파일 읽기
    with open(config_path,"r",encoding="utf-8") as f:
        config = json.load(f)

    # 파일 로드
    train_df = pd.read_csv(config["train"]["output_file"])
    test_df = pd.read_csv(config["test"]["output_file"])
    
    # 레이블 인코딩 ('ham'과 'spam'을 각각 0과 1로 변환)
    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['label'])  # 'ham' → 0, 'spam' → 1
    test_df['label'] = label_encoder.transform(test_df['label'])

    train_texts = train_df['text'].tolist()  # 텍스트 열에서 리스트로 변환
    train_labels = train_df['label'].tolist()  # 레이블 열에서 리스트로 변환

    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()


    # 데이터셋 및 DataLoader 준비
    tokenizer_model = tokenizers_models()
    train_dataset = KoBERTDataset(train_texts, train_labels, tokenizer_model.tokenizer, max_len=Config.MAX_LEN)
    test_dataset = KoBERTDataset(test_texts, test_labels, tokenizer_model.tokenizer, max_len=Config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # input_ids 범위 확인
    for batch in train_loader:
        input_ids, attention_mask, token_type_ids, labels = batch
        print(f"input_ids max value: {torch.max(input_ids)}")
        break


    return train_loader, test_loader, train_dataset, test_dataset, train_labels, test_labels