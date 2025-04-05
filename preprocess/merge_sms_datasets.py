import os
import glob
import pandas as pd

def load_and_label_csvs(folder_path, label, text_col='text'):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"📂 {label} 폴더에서 {len(csv_files)}개 파일을 로드합니다.")

    dfs = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, engine='python', on_bad_lines='skip')
            # HAM: 첫 열만 사용
            if label == 'ham':
                df = df.iloc[:, 0:1]  # 첫 번째 열만
                df.columns = ['text']
                
                df['text'] = df['text'].astype(str).str.replace(r'^\d+\s*[:：]\s*', '', regex=True)

            # SPAM: CN 컬럼 처리
            elif label == 'spam':
                df = pd.read_csv(path, engine='python', on_bad_lines='skip')  # header 존재
                if 'CN' in df.columns:
                    df = df[['CN']].rename(columns={'CN': 'text'})
                else:
                    raise ValueError(f"'CN' column not found in spam file: {path}")

            df['label'] = label
            dfs.append(df)

        except Exception as e:
            print(f"⚠️ {path} 에서 오류 발생: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['text', 'label'])


def main():
    ham_dir = "data/raw/ham/csv"
    spam_dir = "data/raw/spam/csv"
    output_file = "data/spam.csv"

    ham_df = load_and_label_csvs(ham_dir, label="ham")
    spam_df = load_and_label_csvs(spam_dir, label="spam")

    combined = pd.concat([ham_df, spam_df], ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"✅ 병합 완료! 저장 위치: {output_file}")


if __name__ == "__main__":
    main()