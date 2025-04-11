import pandas as pd
import argparse

from sklearn.metrics import classification_report, confusion_matrix


def load_prediction_csv(model_type):
    filename = {
        "bert": "result/bert/pred_bert+revised.csv",
        "electra": "result/electra/pred_electra+revised.csv",
        "roberta": "result/roberta/pred_roberta+revised.csv",
        "bigbird": "result/bigbird/pred_bigbird+revised.csv"
    }.get(model_type.lower())

    if filename is None:
        raise ValueError(f"❌ 지원하지 않는 모델 타입입니다: {model_type}")
    
    print(f"📂 로드한 파일: {filename}")
    return pd.read_csv(filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, help='bert | electra | roberta | bigbird')
    args = parser.parse_args()

    df = load_prediction_csv(args.model_type)
    wrong = df[df['label'] != df['pred']]
    print(f"❗️ 오분류 샘플 수: {len(wrong)}")

    print(classification_report(df['label'], df['pred'], digits=4))
    print(confusion_matrix(df['label'], df['pred']))

if __name__ == "__main__":
    main()