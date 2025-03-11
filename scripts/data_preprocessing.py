import pandas as pd
import re
import json
import os
import glob
from config import Label

def main():
    # 현재 스크립트 위치 기준으로 config.json 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "json/config.json")
    
    # 설정 파일 읽기
    with open(config_path,"r",encoding="utf-8") as f:
        config = json.load(f)

    # ham csv 호출
    ham_csv_files = glob.glob(config["ham_csv"]["input_dir"])
    # spam csv 호출
    spam_csv_files = glob.glob(config["spam"]["input_dir"])
    # 각 전처리 함수 호출
    preprocess_ham(ham_csv_files, config["ham_csv"]["output_file"])
    preprocess_spam(spam_csv_files, config["spam"]["output_file"])
    

# 텍스트 처리 함수 설정
def preprocess_text(text):
    # 불용 키워드
    REMOVE_KEYWORDS = ["Web발신", "웹발신", "web발신", "국제발신","국외발신", "광고", "무료수신거부"]
    
    # 여러 개의 공백을 하나의 공백으로 축소
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()
    # 맨 앞의 숫자와 공백 또는 콜론 제거
    text = re.sub(r'^\d+\s*', '', text)
    # 특수 문자 제거
    # text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    text = re.sub(r"[^\w\s@#%!?]", "", text)  # Retain @, #, %, !, ?
    text = re.sub(r'ifg@', '', text)
    # URL 마스킹
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    # 불용 키워드 제거
    for keyword in REMOVE_KEYWORDS:
        text=text.replace(keyword, "")
    
    return text

# 레이블링 함수
def label_text(text):
    for label, keywords in Label.labels.items():
        if any(keyword in text for keyword in keywords):
            return label
    return "기타"

# csv 파일 전처리 함수
def preprocess_spam(file_paths, output_path):
    """
    spam 데이터셋 전처리 함수
    """
    chunk_size = 50000  # 메모리 제한을 고려하여 적절한 크기 설정
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for file in file_paths:
            for chunk in pd.read_csv(file, chunksize=chunk_size):
                chunk.dropna(inplace=True)
                chunk.columns=['label', 'text']
                chunk.set_index('label')
                
                # 텍스트 전처리 적용
                chunk['text'] = chunk['text'].apply(preprocess_text)
                
                # 기존 'spam'이 아니라, label_text() 함수를 이용해 카테고리별 레이블링
                # chunk['label'] = chunk['text'].apply(label_text)
                chunk['label'] = 'spam'
                
                # 컬럼 순서 조정: 'text' → 'label'
                chunk = chunk[['text', 'label']] 
                
                # 불용 칼럼 제거
                chunk.loc[:, chunk.columns.intersection(['text','label'])]
                
                # 결과 저장
                chunk.to_csv(f_out, mode='a', header=f_out.tell()==0, index=False)


def preprocess_ham(file_paths, output_path):
    """
    ham 데이터셋 전처리 함수
    """
    chunk_size = 50000  # 메모리 제한을 고려하여 적절한 크기 설정
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for file in file_paths:
            for chunk in pd.read_csv(file, chunksize=chunk_size, sep='\t', header=None, names=['raw_text']):
                # ':' 이후의 텍스트만 남기고 전처리
                chunk['text'] = chunk['raw_text'].apply(lambda x: x.split(':', 1)[1].strip() if ':' in str(x) else x)
                chunk.dropna(inplace=True)
                # 불필요한 컬럼 제거
                chunk = chunk[['text']]

                # 텍스트 전처리 적용
                chunk['text'] = chunk['text'].apply(preprocess_text)

                # 라벨링 적용
                chunk['label'] = 'ham'

                # 결과 저장
                chunk.to_csv(f_out, mode='a', header=f_out.tell()==0, index=False)

# xlsx 파일 전처리 함수
def preprocess_ham_xlsx(file_paths, output_path):
    """
    ham 데이터셋 전처리 함수
    """
    # ham 파일 읽기
    dfs = [pd.read_excel(file, engine='openpyxl') for file in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    # NaN값 제거
    df = df.dropna()
    # 컬럼명 추가
    df['label'] = 'ham'
    ### 컬럼명 변경 - 특정 데이터셋 ###
    df = df.rename(columns={'SENTENCE': 'text'})
    # 불용 컬럼 제거
    df = df.loc[:, df.columns.intersection(['text','label'])]
    # 데이터 전처리
    df['text'] = df['text'].apply(preprocess_text)
    df.to_csv(output_path, index=False)
    # 결측치 및 중복값 제거
    drop_index = df[df['text'].isnull()].index
    df = df.drop(drop_index)
    df = df.drop_duplicates('text')
    
    # 저장
    df.to_csv(output_path, index=False)

def preprocess_spam_xlsx(file_paths, output_path):
    """
    spam 데이터셋 전처리 함수
    """
    # spam 파일 읽기
    dfs = [pd.read_excel(file, engine='openpyxl') for file in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    # NaN값 제거
    df = df.dropna()
    # 컬럼명 추가
    df['label'] = 'spam'
    ### 컬럼명 변경 - 특정 데이터셋 ###
    df = df.rename(columns={'SENTENCE': 'text'})
    # 불용 컬럼 제거
    df = df.loc[:, df.columns.intersection(['text','label'])]
    # 데이터 전처리
    df['text'] = df['text'].apply(preprocess_text)
    df.to_csv(output_path, index=False)
    # 스팸 레이블 설정
    # df['label'] = 'spam'
    df['label'] = df['text'].apply(label_text)
    # 결측치 및 중복값 제거
    drop_index = df[df['text'].isnull()].index
    df = df.drop(drop_index)
    df = df.drop_duplicates('text')
    
    # 저장
    df.to_csv(output_path, index=False)

# 메인 함수 설정
if __name__ == "__main__":
    main()