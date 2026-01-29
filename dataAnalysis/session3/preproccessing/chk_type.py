import pandas as pd

def check_type(df):
    dtypes = df.dtypes
    n_unique = df.nunique()
    return pd.DataFrame({
        "Dtype": dtypes,
        "num_unique": n_unique
    }).T
