import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from typing import List, Union
from sklearn.preprocessing import MinMaxScaler



def scale_by_factor(df: pd.DataFrame, cols: list, factor: float) -> pd.DataFrame:
    df[cols] /= factor
    return df

def moving_avarage(df: pd.DataFrame, cols: List[str], window='0.5s') -> pd.DataFrame:
    '''
    Compute moving avarage for "cols" on a time window.
    It is supposed that exists a column 'TIMESTAMP' and a column 'OFFSET_s'
    '''
    df = df.set_index(df['TIMESTAMP'].values)
    df[cols] = df[cols].rolling(window).mean()
    df = df.set_index(df['OFFSET_s'].values)
    return df

def magnitudo(df: pd.DataFrame, cols_xyz: Union[List[str], List[list]], names_magnitudo: Union[str, List[str]]) -> pd.DataFrame:
    '''
    given a list of lists, each one containg the three coordinates of an array, compute magnitudos.
    For cols_xyz you can provide also just a triple of coordinates.
    '''
    cols_xyz = [cols_xyz] if isinstance(cols_xyz[0], str) else cols_xyz
    names_magnitudo = [names_magnitudo] if isinstance(names_magnitudo, str) else names_magnitudo
    for col_xyz, name_magnitudo in zip(cols_xyz, names_magnitudo):
        df[name_magnitudo] = np.sqrt((df[col_xyz]**2).sum(axis=1))
    return df

def integrate_cols(df: pd.DataFrame, cols: List[str], offset_col: str = 'OFFSET_s') -> pd.DataFrame:
    '''
    compute the integral function with trapezoid methods of several cols.
    create in the dataframe the coluns "int_{column}"
    '''
    for c in cols:
        integral_values = np.zeros(df.shape[0])
        integral_values[1:] = cumulative_trapezoid(x=df[offset_col], y=df[c])
        df[f'int_{c}'] = integral_values
    
    return df

def diff_dt(df: pd.DataFrame, cols: list, offset_col: str = 'OFFSET_s') -> pd.DataFrame:
    '''
    compute derivative d(col)/dt for each col in cols.
    We use offset_col as time columns, an the mean of df[offset_col].diff() as dt for stability purpose.
    '''
    for c in cols:
        # df[f'dt_{c}'] = df[c].diff().div(df['OFFSET_s'].diff())
        df[f'dt_{c}'] = df[c].diff() / df[offset_col].diff().iloc[1:].mean()

    df = df.iloc[1:]

    return df

def clip(df: pd.DataFrame, cols: List[str], min_: Union[None, float], max_: Union[None, float]) -> pd.DataFrame:
    '''
    clip the values of cols between a min_ and max_ value.
    '''
    df[cols] = np.clip(df[cols], min_, max_)
    return df

def _angle_beetwen_vectors(values_xyz: np.array) -> np.array:
    '''
    given a N x 3 array rappresenting the 3 coordinates of N vectors,
    compute angles between each vector and the previous.
    The first value is 0.

    return a N x 1 array
    '''
    values_norm = np.sqrt((values_xyz**2).sum(axis=1))
    scalar_product = (values_xyz[1:] * values_xyz[:-1]).sum(axis=1)
    product_norms = values_norm[1:] * values_norm[:1]
    
    cos_angles = np.clip(scalar_product / product_norms, -1, 1)
    _angles = np.arccos(cos_angles)
    angles = np.zeros_like(values_norm)
    angles[1:] = _angles

    return angles

def _cross_vectors(values_xyz: np.array) -> np.array:
    '''
    given a N x 3 array rappresenting the 3 coordinates of N vectors,
    compute the cross product between each vector and the next.
    The first value is [0,0,0].

    return a N x 3 array
    '''
    cross_products = np.zeros_like(values_xyz)
    cross_products[1:] = np.cross(values_xyz[:-1], values_xyz[1:])
    return cross_products

def angle_beetwen_vectors(df: pd.DataFrame, cols_xyz: List[List[str]], angles_names: List[str]) -> pd.DataFrame:
    '''
    given a list of lists, each one containg the three coordinates of an array, compute angles
    between an array and the previous.
    '''
    for col_xyz, angle_name in zip(cols_xyz, angles_names):
        df[angle_name] = _angle_beetwen_vectors(df[col_xyz].values)
    return df

def angle_beetwen_cross_vectors(df: pd.DataFrame, cols_xyz: List[List[str]], angles_names: List[str]) -> pd.DataFrame:
    '''
    given a list of lists, each one containg the three coordinates of an array, compute the cross vecotor
    between an array and the previous one, then compute the angles between the cross vectors.
    '''
    for col_xyz, angle_name in zip(cols_xyz, angles_names):
        cross_vectors = _cross_vectors(df[col_xyz].values)
        angles = np.zeros(cross_vectors.shape[0])
        angles[1:] = _angle_beetwen_vectors(cross_vectors[1:])
        df[angle_name] = angles 

    df = df.iloc[2:]

    return df

def lag_features(df: pd.DataFrame, lag: int = 2, cols: Union[List[str], None] = None) -> pd.DataFrame:
    '''
    compute lag features shifting on index
    '''
    
    if cols is None:
        cols = df.columns.to_list() 

    for i in range(1, lag+1):
        df[[f'{c}_l{i}' for c in cols]] = df[cols].shift(i)
    
    return df

def create_grouping_col_by_second(df: pd.DataFrame, sec: float = 0.25, offset_col: str = 'OFFSET_s') -> pd.DataFrame:
    '''
    compute the column group_{sec}_sec which has integer values.
    If column group_{sec}_sec = n, it means that the row offset_col is in ((n-1)*sec, n*sec] range. 
    '''
    grid = np.arange(df[offset_col].min(), df[offset_col].max()+sec, sec)
    labels = list(range(grid.size - 1))
    df[f'group_{sec}_sec'] = pd.cut(df[offset_col], bins=grid, labels=labels, include_lowest=True)
    return df

def scale(df: pd.DataFrame, cols: Union[List[str], None] = None, scaler: Union[object, None] = None) -> pd.DataFrame:
    '''
    scale features using scikit learn fitted scaler object.
    if scaler is None, a MinMaxScaler is fitted and used as scaler.
    '''
    if cols is None:
        cols = df.columns.to_list()
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df[cols])

    df[cols] = scaler.transform(df[cols])
    return df

def add_hours(df: pd.DataFrame, hours: int, col_timestamp: str = 'TIMESTAMP') -> pd.DataFrame:
    '''
    add some hours to timestamp column.
    Usefull to syncronize csv to video
    '''
    df[col_timestamp] += pd.Timedelta(hours=2)
    return df