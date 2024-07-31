from typing import List, Union, Tuple
from functools import reduce 
import pandas as pd


def read_csv_ganci(path_csv: str) -> pd.DataFrame:
    '''
    read csv skipping with correct header and sep
    '''
    return pd.read_csv(
        path_csv,
        header=6,
        sep=';',
    )

def csv_shift(df: pd.DataFrame) -> pd.DataFrame:
    '''
    csv files have one separetor ";" at the end. Columns headers must be shifted
    '''
    df_res = pd.DataFrame()
    df_res['PAN'] = df.index
    idx_col = enumerate(df.columns)
    _ = idx_col.__next__() # to skip first iteration
    for idx, c in idx_col:
        df_res[c] = df.iloc[:, idx - 1].values

    return df_res

def split_sensors(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_an = df[[c for c in df.columns if 'AN' in c] + ['TIMESTAMP']]
    del df_an['PAN']
    # df_an['type'] = 'AN'
    df_an = df_an.rename(columns={c: c.replace('AN', '') for c in df_an.columns})
    
    df_m1 = df[[c for c in df.columns if c.endswith('TM1')] + ['TIMESTAMP']] 
    # df_m1['type'] = 'M1'  
    df_m1 = df_m1.rename(columns={c: c.replace('TM1', '') for c in df_m1.columns})
    
    df_r1 = df[[c for c in df.columns if c.endswith('TR1')] + ['TIMESTAMP']] 
    # df_r1['type'] = 'R1'  
    df_r1 = df_r1.rename(columns={c: c.replace('TR1', '') for c in df_r1.columns})

    df_m2 = df[[c for c in df.columns if c.endswith('TM2')] + ['TIMESTAMP']] 
    # df_m2['type'] = 'M2'  
    df_m2 = df_m2.rename(columns={c: c.replace('TM2', '') for c in df_m2.columns})

    df_r2 = df[[c for c in df.columns if c.endswith('TR2')] + ['TIMESTAMP']] 
    # df_r2['type'] = 'R2'  
    df_r2 = df_r2.rename(columns={c: c.replace('TR2', '') for c in df_r2.columns})

    # return df_an, df_m1, df_m2, df_r1, df_r2
    return {
        'AN': df_an,
        'TM1': df_m1,
        'TM2': df_m2,
        'TR1': df_r1,
        'TR2': df_r2
    }

def timestamp(df: pd.DataFrame) -> pd.DataFrame:
    '''
    - convert TIMESTAMP in datetime
    - computing OFFSET_ms (millisecond)
    - computing OFFSET_s (second)
    - set OFFSET_s as index;
    - order by OFFSET_s;
    '''
    df['OFFSET_ms'] = df['TIMESTAMP'] - df['TIMESTAMP'].iloc[0]
    df['OFFSET_s'] = df['OFFSET_ms'] / 1_000
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'].values, unit='ms')

    df = df.set_index(df['OFFSET_s'].values)
    df = df.sort_index()

    return df

def drop_data_by_lower_bounds(df: pd.DataFrame, cols: Union[List[str], str], lower_bounds: Union[List[float], float]) -> pd.DataFrame:
    '''
    given a list of cols and a corrispondent list of lower bounds, discard rows which are under the
    threshold for some one columns.
    you can provide str for cols, float for lower_bounds if you want just one filter.
    '''
    cols = [cols] if isinstance(cols, str) else cols
    lower_bounds = [lower_bounds] if isinstance(lower_bounds, str) else lower_bounds
    
    masks = []
    for col, cut in zip(cols, lower_bounds):
        masks.append(
            df[col] > cut
        )
    mask = reduce(lambda m1, m2: m1 & m2, masks)
    return df[mask]

def keep_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df[cols]

def reset_index_OFFSET(df: pd.DataFrame) -> pd.DataFrame:
    df.index -= df.index[0]
    return df