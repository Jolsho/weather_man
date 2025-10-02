import numpy as np
import torch

numeric_cols = [
    "temp", "dewPt", "rh", "pressure", "wspd", "uv_index",
    "vis", "heat_index", "wc", "feels_like", "precip_hrly",
    "temp_diff_1", "temp_diff_3",
    "temp_roll_mean_3", "temp_roll_std_6",
    "temp_slope_6", "temp_slope_norm"
]


def analyze_residuals(model, loader, dataset, device, feature_cols, target_df):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            preds.append(pred)
            trues.append(y_batch.numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    scaler = dataset.scalers[dataset.target_city_index]

    def inverse_temp(vals):
        dummy = np.zeros((len(vals), len(dataset.scalers[0].mean_)))
        temp_index = numeric_cols.index("temp")
        dummy[:, temp_index] = vals
        return scaler.inverse_transform(dummy)[:, temp_index]

    # Residuals at horizon t+1
    residuals = inverse_temp(y_true[:, 0]) - inverse_temp(y_pred[:, 0])

    # Align with features from target city's dataframe
    df = target_df.iloc[-len(residuals):].copy()
    df["residuals"] = residuals

    # Correlation with some new engineered features
    for f in feature_cols:
        corr = df[f].corr(df["residuals"])
        print(f"{f}: corr={corr:.3f}")


def add_cyclical_time_features(df):
    # Hour cyclical
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Month cyclical
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df
