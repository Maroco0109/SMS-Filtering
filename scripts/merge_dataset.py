import pandas as pd
import json
import os
import glob

def merge_datasets(input_path_pattern, output_path):
    files = glob.glob(input_path_pattern)
    combined = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    combined.drop_duplicates(subset=['text'], inplace=True)
    combined.dropna(subset=["text"], inplace=True)    # NaN값 제거
    combined["text"]=combined["text"].astype(str)   # text 값을 str로 강제 변환
    
    combined.to_csv(output_path, index=False)
    
def main():
    # 현재 스크립트 위치 기준으로 config.json 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "json/config.json")
    
    # 설정 파일 읽기
    with open(config_path,"r",encoding="utf-8") as f:
        config = json.load(f)
        
    # 데이터 병합
    input_path_pattern = config["sms-data"]["input_dir"]
    output_path = config["sms-data"]["output_file"]
    
    merge_datasets(input_path_pattern=input_path_pattern, output_path=output_path)

# 메인 함수 설정
if __name__ == "__main__":
    main()