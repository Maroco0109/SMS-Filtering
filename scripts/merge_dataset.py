import pandas as pd
from glob import glob

def merge_datasets(input_path_pattern, output_path):
    files = glob(input_path_pattern)
    combined = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    combined.drop_duplicates(subset=['text'], inplace=True)
    combined.to_csv(output_path, index=False)