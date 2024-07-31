import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def read_csv_ganci(file_path):
    df_ = pd.read_csv(
        file_path,
        header=6,
        sep=';',
    )

    df = pd.DataFrame()
    df['PAN'] = df_.index
    for i, c in enumerate(df_.columns):
        if i > 0:
            df[c] = df_.iloc[:, i - 1].values

    df = df.set_index(pd.to_datetime(df.TIMESTAMP, unit='ms'))
    df = df.sort_index()
    
    return df
