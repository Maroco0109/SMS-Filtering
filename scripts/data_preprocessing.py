import pandas as pd
import re

# 텍스트 처리 함수 설정
def preprocess_text(text):
    # 여러 개의 공백을 하나의 공백으로 축소
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()
    # 맨 앞의 숫자와 공백 또는 콜론 제거
    text = re.sub(r'^\d+\s*', '', text)
    # 필요에 따라 특수 문자 제거 (예시로 일부 문자만 제거)
    # text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    text = re.sub(r"[^\w\s@#%!?]", "", text)  # Retain @, #, %, !, ?
    # 필요에 따라 특수 문자 제거 (예시로 일부 문자만 제거)
    text = re.sub(r'ifg@', '', text)
    # URL 마스킹
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    # *이 두개 이상 발생할 시 삭제
    text = re.sub(r"\*{2,}","", text)
    
    return text

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
    df['label'] = 'spam'
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
    # 컬럼명 변경 - 특정 데이터셋
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