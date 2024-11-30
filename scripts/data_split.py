import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(data_path, output_train, output_test, test_size=0.2, random_state=0):
    data = pd.read_csv(data_path)
    X = data['text']
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    
    train_set.to_csv(output_train, index=False)
    test_set.to_csv(output_test, index=False)