import pandas as pd
import os
import json
import glob
from sklearn.model_selection import train_test_split

def split_dataset(data_path, output_train, output_test, test_size=0.2, random_state=0):
    data = pd.read_csv(data_path)
    
    X = data['text']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    
    # 출력 디렉토리 자동 생성
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    os.makedirs(os.path.dirname(output_test), exist_ok=True)
    
    # 파일 저장
    train_set.to_csv(output_train, index=False)
    test_set.to_csv(output_test, index=False)
    
def main():
    # 현재 스크립트 위치 기준으로 config.json 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "json/config.json")
    
    # 설정 파일 읽기
    with open(config_path,"r",encoding="utf-8") as f:
        config = json.load(f)
        
    # 데이터 분할
    data_path = config["sms-data"]["output_file"]
    output_train = config["train"]["output_file"]
    output_test = config["test"]["output_file"]
    split_dataset(data_path, output_train, output_test)
    
# 메인 함수 설정
if __name__ == "__main__":
    main()