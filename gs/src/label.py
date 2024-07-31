import pandas as pd

def label(df: pd.DataFrame, df_label: pd.DataFrame) -> pd.DataFrame:
    """
    add the TYPE_EVENT values to df as columns in onehot enconding way

    Args:
        df (pd.DataFrame): dataframe having the column OFFSET_s
        df_label (pd.DataFrame): dataframe having columns TYPE_EVENT,START_OFFSET_SECOND,END_OFFSET_SECOND

    Returns:
        pd.DataFrame: return labeled dataframe
    """
    labels = df_label['TYPE_EVENT'].unique()
    for label in labels:
        filtered = df_label.query(f'TYPE_EVENT == "{label}"')
        intervals = list(zip(filtered['START_OFFSET_SECOND'], filtered['END_OFFSET_SECOND']))
        bins = pd.IntervalIndex.from_tuples(intervals)

        is_event = pd.cut(df['OFFSET_s'], bins=bins)
        df[label] = 0
        df.loc[is_event.notna(), label] = 1

    return df