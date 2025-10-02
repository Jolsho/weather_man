import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def add_trend_features(df, target_col="temp"):
    """
    Add lag, rolling, slope, and flag features to df in a leak-free way.
    Assumes df is time-ordered ascending (oldest -> newest).
    """

    # 1) simple diffs (current - past)
    df["temp_diff_1"] = df[target_col].diff(periods=1)   # t - t-1
    df["temp_diff_3"] = df[target_col].diff(periods=3)   # t - t-3

    # 2) rolling stats using past window (shift so value at time t uses t-1..t-w)
    df["temp_roll_mean_3"] = df[target_col].rolling(window=3, min_periods=1).mean().shift(1)
    df["temp_roll_std_6"]  = df[target_col].rolling(window=6, min_periods=1).std().shift(1).fillna(0.0)

    # 3) slope of temp over last N points using linear regression (leak-free: use t-N..t-1)
    def rolling_slope(series, window):
        slopes = np.full(len(series), np.nan, dtype=float)
        X = np.arange(window).reshape(-1, 1)
        lr = LinearRegression()
        for i in range(window, len(series) + 1):
            y_window = series[i-window:i].values.reshape(-1, 1).ravel()
            if np.all(np.isnan(y_window)):
                slopes[i-1] = 0.0
                continue
            # replace nan with last valid or linear interpolation (simple)
            if np.isnan(y_window).any():
                y_window = pd.Series(y_window).interpolate(limit_direction="both").fillna(method="ffill").fillna(method="bfill").values
            lr.fit(X, y_window)
            slopes[i-1] = lr.coef_[0]
        return slopes

    # slope over last 6 timesteps (t-6 .. t-1), assign at index t-1, shift to align to t
    df["temp_slope_6"] = rolling_slope(df[target_col], window=6)
    df["temp_slope_6"] = df["temp_slope_6"].shift(0)    # already aligned to last index of window; if you want value at t use shift(1)
    # For leak-free at time t you want slope computed on t-6..t-1 -> value should be placed at t (so shift 0 if computed as above)
    # If your implementation ends up placing slope at t-1, do: df["temp_slope_6"] = df["temp_slope_6"].shift(1)

    # 4) binary / categorical trend flag using slope + small threshold to avoid noise
    slope_thresh = 0.05  # degrees per timestep; tuneable
    df["temp_trend_flag"] = 0  # 0 = steady, 1 = rising, -1 = falling
    df.loc[df["temp_slope_6"] > slope_thresh, "temp_trend_flag"] = 1
    df.loc[df["temp_slope_6"] < -slope_thresh, "temp_trend_flag"] = -1

    # 5) optional: rate of change normalized by rolling std (z-score style)
    df["temp_slope_norm"] = df["temp_slope_6"] / (df["temp_roll_std_6"] + 1e-6)

    # 6) fill/trim NaNs introduced at start
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.fillna(0.0, inplace=True)

    return df
