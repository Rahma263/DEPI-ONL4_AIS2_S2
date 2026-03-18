import pandas as pd
def is_Null(df):
   null=df.isnull().sum()
   return pd.DataFrame(null).T