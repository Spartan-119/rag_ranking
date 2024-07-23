# data_loader.py

import pandas as pd

def load_job_descriptions(file_path, column_index=0):
    # Read CSV without header, use the first column by default
    df = pd.read_csv(file_path, header=None)
    print(f"The CSV file has {df.shape[1]} columns.")
    
    if column_index >= df.shape[1]:
        raise ValueError(f"Column index {column_index} is out of range. The file only has {df.shape[1]} columns.")
    
    return df[column_index].tolist()

def load_kpi_document(file_path):
    with open(file_path, 'r') as file:
        return file.read()