import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_split(filepath, train_ratio=0.7, val_ratio=0.1):
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    values = df.values.astype(np.float32)
    n = len(values)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return values[:t1], values[t1:t2], values[t2:]

def normalize(train, val, test):
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train)
    val_s   = scaler.transform(val)
    test_s  = scaler.transform(test)
    return train_s, val_s, test_s, scaler
