import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import other

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Using device: {device}")

# ---------------- Features ----------------
features = [
    "temp", "dewPt", "rh", "pressure",
    "wdir", "wspd", "uv_index", "vis",
    "heat_index", "wc", "feels_like",
    "precip_hrly", "clds", "wdir_cardinal",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # "temp_diff_1", "temp_diff_3",
    # "temp_roll_mean_3", "temp_roll_std_6",
    # "temp_slope_6", "temp_trend_flag", "temp_slope_norm"
]

# Numeric columns for scaling
numeric_cols = [
    "temp", "dewPt", "rh", "pressure", "wspd", "uv_index",
    "vis", "heat_index", "wc", "feels_like", "precip_hrly",
    # "temp_diff_1", "temp_diff_3",
    # "temp_roll_mean_3", "temp_roll_std_6",
    # "temp_slope_6", "temp_slope_norm"
]


# ---------------- Dataset ----------------
class WeatherDataset(Dataset):
    def __init__(self, df_list, target_city_index, seq_len=3, features=None, numeric_cols=None):
        self.seq_len = seq_len
        self.target_city_index = target_city_index

        # Validate and select features
        for i, df in enumerate(df_list):
            if features:
                missing_cols = [f for f in features if f not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in city index {i}: {missing_cols}")
                df_list[i] = df[features].copy()

        # Scale numeric columns
        self.scalers = []
        for i, df in enumerate(df_list):
            scaler = StandardScaler()
            if numeric_cols:
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers.append(scaler)

        self.num_features = df_list[0].shape[1]

        # Stack data
        self.X = pd.concat([df for i, df in enumerate(df_list)], axis=1).values
        self.y_all = df_list[target_city_index]['temp'].values.astype(np.float32)

        self.max_horizon = 24
        self.length = len(self.X) - seq_len - self.max_horizon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        end_idx = idx + self.seq_len
        if end_idx + 5 >= len(self.y_all):
            raise IndexError(f"Index {idx} out of range for sequence length {self.seq_len}")
        X_seq = self.X[idx:end_idx]
        y_targets = np.array([
            self.y_all[end_idx + 0],  # t+1
            self.y_all[end_idx + 2],  # t+3
            self.y_all[end_idx + 5],  # t+6
            self.y_all[end_idx + 11],  # t+12
            self.y_all[end_idx + 23],  # t+24
        ], dtype=np.float32)
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.float32)


# ---------------- Model ----------------
class TempPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, output_size=5, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


# ---------------- Load Data ----------------
cities = ["Kingman", "Henderson", "Nellis", "Vegas", "IndianSprings"]
df_list = [pd.read_csv(f"data/{city}.csv") for city in cities]
# Add engineered features
for i in range(len(df_list)):
    df_list[i] = other.add_cyclical_time_features(df_list[i])

target_city_index = 3
seq_len = 24

dataset = WeatherDataset(df_list, target_city_index, seq_len, features=features, numeric_cols=numeric_cols)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TempPredictor(input_size=len(cities)*len(features)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()


# ---------------- Train/Val/Test Split ----------------
def split_dataset(dataset, train_frac=0.7, val_frac=0.15):
    n = len(dataset)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)
    idx_train = range(0, train_end)
    idx_val = range(train_end, val_end)
    idx_test = range(val_end, n)

    return torch.utils.data.Subset(dataset, idx_train), \
        torch.utils.data.Subset(dataset, idx_val), \
        torch.utils.data.Subset(dataset, idx_test)


train_set, val_set, test_set = split_dataset(dataset)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# ---------------- Training Loop with Validation ----------------
num_epochs = 10
torch.manual_seed(42)
np.random.seed(42)
best_val_loss = float("inf")
patience, patience_counter = 5, 0

for epoch in range(num_epochs):
    # ----- Training -----
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    # ----- Validation -----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ----- Early stopping -----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best weights
model.load_state_dict(torch.load("best_model.pt"))


# ---------------- Evaluation Function (with inverse scaling) ----------------
def inverse_transform_temp(scaler, values):
    """Inverse transform standardized temps to original units."""
    # Make a temp-only DataFrame since scaler expects all numeric cols
    dummy = np.zeros((len(values), len(numeric_cols)))
    temp_index = numeric_cols.index("temp")
    dummy[:, temp_index] = values
    return scaler.inverse_transform(dummy)[:, temp_index]


def evaluate_model(model, loader, dataset, device, horizons=[1, 3, 6, 12, 24]):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Inverse transform back to degrees
    scaler = dataset.scalers[dataset.target_city_index]
    for i, h in enumerate(horizons):
        y_true_deg = inverse_transform_temp(scaler, y_true[:, i])
        y_pred_deg = inverse_transform_temp(scaler, y_pred[:, i])

        rmse = np.sqrt(mean_squared_error(y_true_deg, y_pred_deg))
        mae = mean_absolute_error(y_true_deg, y_pred_deg)
        r2 = r2_score(y_true_deg, y_pred_deg)

        print(f"Horizon t+{h}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")


def evaluate_baseline(dataset, horizons=[1, 3, 6, 12, 24]):
    """
    Baseline: predict future temp as yesterday's temp at the same hour.
    """
    y_true, y_pred = [], []

    # Grab the target city temps (already scaled)
    temps = dataset.y_all

    for i in range(len(dataset)):
        _, y_targets = dataset[i]  # ground truth temps (already aligned to horizons)

        # Compute baseline: previous day's temps at same horizon offsets
        # Here we assume hourly data
        baseline_preds = []
        for h in horizons:
            prev_day_idx = i + dataset.seq_len + (h - 24)  # 24 hours before
            if prev_day_idx < 0:
                baseline_preds.append(np.nan)  # skip if not enough history
            else:
                baseline_preds.append(temps[prev_day_idx])
        y_true.append(y_targets.numpy())
        y_pred.append(np.array(baseline_preds))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Drop rows with NaNs (early indices with no baseline available)
    mask = ~np.isnan(y_pred).any(axis=1)
    y_true, y_pred = y_true[mask], y_pred[mask]

    # Inverse transform back to degrees
    scaler = dataset.scalers[dataset.target_city_index]
    for i, h in enumerate(horizons):
        y_true_deg = inverse_transform_temp(scaler, y_true[:, i])
        y_pred_deg = inverse_transform_temp(scaler, y_pred[:, i])

        rmse = np.sqrt(mean_squared_error(y_true_deg, y_pred_deg))
        mae = mean_absolute_error(y_true_deg, y_pred_deg)
        r2 = r2_score(y_true_deg, y_pred_deg)

        print(f"[Baseline] Horizon t+{h}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")


# ---------------- Run Final Evaluation ----------------
print("\nValidation performance (model):")
evaluate_model(model, val_loader, dataset, device)

print("\nValidation performance (baseline):")
evaluate_baseline(dataset)

print("\nTest performance (model):")
evaluate_model(model, test_loader, dataset, device)

print("\nTest performance (baseline):")
evaluate_baseline(dataset)

# Example single prediction in degrees
X_sample, y_true_sample = test_set[0]
X_sample = X_sample.unsqueeze(0).to(device)
y_pred_sample = model(X_sample).detach().cpu().numpy()[0]

scaler = dataset.scalers[dataset.target_city_index]
y_pred_deg = inverse_transform_temp(scaler, y_pred_sample)
y_true_deg = inverse_transform_temp(scaler, y_true_sample.numpy())

# Say you engineered these:
# extra_feats = ["temp_diff_1", "temp_slope_6", "temp_trend_flag"]
# # Use the target city’s dataframe (Vegas in your case)
# vegas_df = pd.read_csv("data/Vegas.csv")
# other.analyze_residuals(model, val_loader, dataset, device, extra_feats, vegas_df)

print("\nSample prediction (°C):")
print(f"Predicted (t+1, t+3, t+6, t+12, t+24): {y_pred_deg}")
print(f"True      (t+1, t+3, t+6, t+12, t+24): {y_true_deg}")
