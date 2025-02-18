import pandas as pd

train_data = pd.read_csv("/home/maroco/SMS-Filtering/data/preprocessed/train.csv")
test_data = pd.read_csv("/home/maroco/SMS-Filtering/data/preprocessed/test.csv")

print(train_data['text'].isna().sum(), train_data['text'].dtype)
print(test_data['text'].isna().sum(), test_data['text'].dtype)
nan_rows=train_data[train_data["text"].isna()]
print(nan_rows)