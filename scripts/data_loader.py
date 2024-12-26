import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from scripts.model import tokenizers_models
from scripts.KoBERTDataset import KoBERTDataset
from scripts.config import Config

# Data loaders
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

    # 레이블 인코딩 ('ham'과 'spam'을 각각 0과 1로 변환)
    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['label'])  # 'ham' → 0, 'spam' → 1
    test_df['label'] = label_encoder.transform(test_df['label'])

    # 데이터셋 및 DataLoader 준비
    train_dataset = KoBERTDataset(train_texts, train_labels, tokenizers_models.tokenizer, max_len=Config.MAX_LEN)
    test_dataset = KoBERTDataset(test_texts, test_labels, tokenizers_models.tokenizer, max_len=Config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_dataset, test_dataset, train_labels, test_labels