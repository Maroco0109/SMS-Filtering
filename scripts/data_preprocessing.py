import pandas as pd
import re

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s@#%!?]', '', text)
    return text

def preprocess_and_label(file_paths, output_path, label):
    """
    데이터 전처리 및 레이블 추가 함수.
    """
    dfs = []
    for file in file_paths:
        if file.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        elif file.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.endswith('.txt'):
            df = pd.read_csv(file, sep='\t', header=None, names=['text'])
        else:
            raise ValueError(f"Unsupported file format: {file}")
        
        # 텍스트 전처리
        df['text'] = df['text'].apply(preprocess_text)
        
        # 레이블 추가
        df['label'] = label
        dfs.append(df)
    
    # 데이터 병합
    result = pd.concat(dfs, ignore_index=True)
    
    # 저장
    result.to_csv(output_path, index=False)

def preprocess_spam(file_paths, output_path):
    dfs = [pd.read_csv(file) for file in file_paths]
    df = pd.concat(dfs)
    df['text'] = df['text'].apply(preprocess_text)
    df.to_csv(output_path, index=False)

def preprocess_ham(file_paths, output_path):
    dfs = [pd.read_csv(file, sep='\t') for file in file_paths]
    df = pd.concat(dfs)
    df['text'] = df['text'].apply(preprocess_text)
    df.to_csv(output_path, index=False)

def preprocess_ham_xlsx(file_paths, output_path):
    dfs = [pd.read_excel(file, engine='openpyxl') for file in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    df['text'] = df['text'].apply(preprocess_text)
    df.to_csv(output_path, index=False)
