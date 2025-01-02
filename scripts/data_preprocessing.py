import pandas as pd
import re
import json
from scripts.config import Label

def main():
    # 설정 파일 읽기
    with open("/json/config.json","r") as f:
        config = json.load(f)

    # 각 전처리 함수 호출
    preprocess_ham(config["ham_csv"]["input_dir"], config["ham_csv"]["output_file"])
    preprocess_ham_xlsx(config["ham_xlsx"]["input_dir"], config["ham_xlsx"]["output_file"])
    preprocess_spam(config["spam"]["input_dir"], config["spam"]["output_file"])
    preprocess_spam_xlsx(config["spam_xlsx"]["input_dir"], config["spam_xlsx"]["output_file"])

# 텍스트 처리 함수 설정
def preprocess_text(text):
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
    # spam 파일 읽기
    dfs = [pd.read_csv(file) for file in file_paths]
    df = pd.concat(dfs)
    # NaN값 제거
    df = df.dropna()
    # 컬럼명 추가
    df.columns=['label', 'text']
    df = df.set_index('label')
    # 스팸 레이블 설정
    # df['label'] = 'spam'
    df['label'] = df['text'].apply(label_text)
    # 불용 컬럼 제거
    df = df.loc[:, df.columns.intersection(['text','label'])]
    # 데이터 전처리
    df['text'] = df['text'].apply(preprocess_text)
    # 결측치 및 중복값 제거
    drop_index = df[df['text'].isnull()].index
    df = df.drop(drop_index)
    df = df.drop_duplicates('text')

    # 저장
    df.to_csv(output_path, index=False)

def preprocess_ham(file_paths, output_path):
    """
    ham 데이터셋 전처리 함수
    """
    # ham 파일 읽기
    dfs = [pd.read_csv(file, sep='\t') for file in file_paths]
    df = pd.concat(dfs)
    # NaN값 제거
    df = df.dropna()
    # 컬럼명 추가
    df['label'] = 'ham'
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

# def preprocess_and_label(file_paths, output_path, label):
#     """
#     데이터 전처리 및 레이블 추가 함수.
#     """
#     dfs = []
#     for file in file_paths:
#         if file.endswith('.xlsx'):
#             df = pd.read_excel(file, engine='openpyxl')
#         elif file.endswith('.csv'):
#             df = pd.read_csv(file)
#         elif file.endswith('.txt'):
#             df = pd.read_csv(file, sep='\t', header=None, names=['text'])
#         else:
#             raise ValueError(f"Unsupported file format: {file}")
#         
#         # 텍스트 전처리
#         df['text'] = df['text'].apply(preprocess_text)
#         
#         # 레이블 추가
#         df['label'] = label
#         dfs.append(df)
#     
#     # 데이터 병합
#     result = pd.concat(dfs, ignore_index=True)
#     
#     # 저장
#     result.to_csv(output_path, index=False)