import pandas as pd
import re

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s@#%!?]', '', text)
    return text

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
