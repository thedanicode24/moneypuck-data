import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(df, column):
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[[column]])
    cols = [f"{column}_{cat}" for cat in encoder.categories_[0]]
    df_encoded = pd.DataFrame(encoded, columns=cols, index=df.index)
    return pd.concat([df.drop(columns=[column]), df_encoded], axis=1)