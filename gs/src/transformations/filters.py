import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from typing import List


def _kalman_filter(df: pd.DataFrame, col: str, dt: float = 0.5, save_original: bool = False) -> pd.DataFrame:
    '''
    apply kalman filter to one column
    '''
    if save_original:
        df[f'original_{col}'] = df[col].copy()

    dim_x = 2
    dim_z = 1

    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.F = np.array([[1, dt], # transizione dello stato
                     [0, 1]])
    kf.H = np.array([[1, 0]]) # Osservazione
    kf.Q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=1e-5) # rumore di processo
    kf.R = np.array([[0.1**2]]) # rumore di misura
    kf.P = np.array([[1, 0], # covarianza iniziaòe
                     [0, 1]])

    values = df[col].values
    filtered = [values[0]]
    kf.x = [values[0], 0]
    for value in values[1:]:
        kf.predict()
        kf.update([value])
        filtered.append(kf.x[0])
    df[col] = filtered
    return df

def kalman_filter(df: pd.DataFrame, cols: List[str], dt: float = 0.5, save_original: bool = False) -> pd.DataFrame:
    '''
    apply kalman filter to multiple columns
    '''
    for c in cols:
        df = _kalman_filter(df=df, col=c, dt=dt, save_original=save_original)
    return df

def _lowess_filter(df: pd.DataFrame, col: str, num_fit: int = 25, save_original: bool = False) -> pd.DataFrame:
    if save_original:
        df[f'original_{col}'] = df[col].copy()

    x = df['OFFSET_s'].values
    y = df[col].values

    frac = num_fit / y.size  # Percentuale dei dati utilizzati per ciascun fit locale
    y_smoth = lowess(y, x, frac=frac)[:, 1]
    df[col] = y_smoth

    return df

def lowess_filter(df: pd.DataFrame, cols: List[str], num_fit: int = 25, save_original: bool = False) -> pd.DataFrame:
    for c in cols:
        df = _lowess_filter(df=df, col=c, num_fit=num_fit, save_original=save_original)
    return df

def _single_point_filter(df: pd.DataFrame, col: str, margin: int = 4) -> pd.DataFrame:
    '''
    set to 0 the values of col which have a neighbourhood of zeros.
    The neighbourhood is defined by margin. 
    '''
    values = df[col].values
    for i in range(values.size):
        left_margin = max(i - margin, 0)
        right_margin = min(i + margin, values.size)
        if (values[left_margin: right_margin] > 0).sum() == 1:
            values[i] = 0

    df[col] = values
    return df

def single_point_filter(df: pd.DataFrame, cols: List[str], margin: int = 4) -> pd.DataFrame:
    '''
    set to 0 the values of cols which have a neighbourhood of zeros.
    The neighbourhood is defined by margin. 
    '''
    for c in cols:
        df = _single_point_filter(df, col=c, margin=margin)
    return df