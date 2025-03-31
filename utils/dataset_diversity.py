import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def evaluate_diversity(csv_path, text_col='proc_text', label_col='label'):
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    spam_count = (df[label_col] == 0).sum()
    ham_count = (df[label_col] == 1).sum()

    # 문장 길이
    df['len'] = df[text_col].apply(lambda x: len(str(x).split()))
    avg_len = df['len'].mean()
    spam_len = df[df[label_col] == 0]['len'].mean()
    ham_len = df[df[label_col] == 1]['len'].mean()

    # Vocabulary Size
    all_tokens = ' '.join(df[text_col].astype(str)).split()
    vocab_size = len(set(all_tokens))

    # n-gram 중복도 (bi-gram)
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    ngram_counts = vectorizer.fit_transform(df[text_col].astype(str))
    total_ngrams = ngram_counts.sum()
    unique_ngrams = len(vectorizer.vocabulary_)
    ngram_dup_ratio = 1 - (unique_ngrams / total_ngrams)

    print(f"\n📄 Dataset Overview")
    print(f"- 총 샘플 수: {total_samples}")
    print(f"- 스팸 샘플 수: {spam_count}")
    print(f"- 햄 샘플 수: {ham_count}")

    print(f"\n📊 다양성 지표")
    print(f"- 전체 고유 단어 수: {vocab_size}")
    print(f"- 평균 문장 길이: {avg_len:.2f} 토큰")
    print(f"- 스팸 평균 길이: {spam_len:.2f}, 햄 평균 길이: {ham_len:.2f}")
    print(f"- 2-gram 중복도: {ngram_dup_ratio*100:.2f}%")

if __name__ == '__main__':
    csv_path = '/home/maroco/SMS-Filtering/data/proc_data.csv'  # 또는 train/valid csv
    evaluate_diversity(csv_path)
