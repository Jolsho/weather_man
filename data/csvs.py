import numpy as np
import pandas as pd
from structs import windDirs, clds, feature_cols, numeric_cols, categorical_cols


def map_categorical_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts categorical columns in the dataframe to their corresponding index
    in windDirs or clds lists.
    Prints unknown values and continues.
    """
    df = df.copy()

    # Mapping dictionaries
    windDirs_map = {val: idx for idx, val in enumerate(windDirs)}
    clds_map = {val: idx for idx, val in enumerate(clds)}

    # Helper function
    def map_column(series, mapping, col_name):
        def mapper(x):
            if x not in mapping:
                print(f"Unknown value in column '{col_name}': {x}")
                return x  # Keep the unknown value as-is
            return mapping[x]
        return series.map(mapper)

    if "wdir_cardinal" in df.columns:
        col = df["wdir_cardinal"]
        df["wdir_cardinal"] = map_column(col, windDirs_map, "wdir_cardinal")
    if "clds" in df.columns:
        df["clds"] = map_column(df["clds"], clds_map, "clds")

    return df


def process_time_features(df, time_col="datetime", drop_original=True):
    """
    Takes a dataframe with a timestamp column and adds:
      - month: month of the year (1–12)
      - hour: hour of day (0–23)
    Optionally drops the original timestamp column.
    """
    # Ensure datetime type
    df[time_col] = pd.to_datetime(df[time_col])

    # Sort by time to be safe
    df = df.sort_values(time_col)

    # Extract month and hour
    df["year"] = df[time_col].dt.year
    df["month"] = df[time_col].dt.month
    df["day"] = df[time_col].dt.day
    df["hour"] = df[time_col].dt.hour

    # Define the columns you want on the left
    left_cols = ["year", "month", "day", "hour"]
    # Reorder the DataFrame
    df = df[left_cols + [c for c in df.columns if c not in left_cols]]

    # Optionally drop original timestamp column
    if drop_original:
        df = df.drop(columns=[time_col])

    return df


def collapse_by_hour(df):
    """
    Collapse rows in a DataFrame by year, month, day, hour.
    For each hour, averages the numeric columns and keeps one row per hour.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'year', 'month', 'day', 'hour'
        and numeric columns. numeric_cols (list of str): Columns to average for
        each hour.

    Returns:
        pd.DataFrame: Collapsed DataFrame with one row per hour.
    """
    # Group by the hour
    grouped = df.groupby(["year", "month", "day", "hour"], as_index=False)

    # Aggregate: mean for numeric, first for categorical
    agg_dict = {col: 'mean' for col in numeric_cols}
    agg_dict.update({col: 'first' for col in categorical_cols})

    df_collapsed = grouped.agg(agg_dict)
    df_collapsed[numeric_cols] = df[numeric_cols].round().astype(int)

    return df_collapsed


def delete_unaligned_dates(dfs):
    """
    Keep only rows with (year, month, day, hour) that exist in all DataFrames.

    Parameters:
        dfs (dict): dictionary of DataFrames, one per dataset

    Returns:
        dict: cleaned and aligned DataFrames
    """
    # Compute the set of hours for each DataFrame
    sets_of_hours = [set(zip(df['year'], df['month'], df['day'], df['hour'])) for df in dfs.values()]

    # Keep only the intersection (hours present in all datasets)
    common_hours = set.intersection(*sets_of_hours)

    # Filter each DataFrame
    aligned_dfs = {}
    for k, df in dfs.items():
        mask = df.set_index(['year','month','day','hour']).index.isin(common_hours)
        aligned_dfs[k] = df[mask].reset_index(drop=True)

    return aligned_dfs
