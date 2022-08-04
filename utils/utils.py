import pandas as pd


def extract_features(df):
    """
    Extract non-zero features from loading matrix.
    """
    tmp = []
    for col in df:
        idx = ~(df[col] == 0)
        tmp = tmp + df.loc[idx, col].index.values.tolist()
    return set(tmp)

