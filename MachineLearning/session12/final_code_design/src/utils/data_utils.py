import pandas as pd

def read_and_clean_data(path):
    df = pd.read_csv(path)
    drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    if 'Age' in df.columns:
        df = df[df['Age'] <= 80].copy()
    return df

