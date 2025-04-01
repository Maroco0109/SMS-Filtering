import pandas as pd

# 디렉토리 수정
df = pd.read_csv('result/pred_electra+revised.csv')

wrong = df[df['label'] != df['pred']]
print(f"❗️ 오분류 샘플 수: {len(wrong)}")

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(df['label'], df['pred'], digits=4))
print(confusion_matrix(df['label'], df['pred']))