import json
import pandas as pd
import os
import csvs
from structs import cities, years, months, cols, categorical_cols, numeric_cols
# import trendy


class page:
    def __init__(self, city, year, month):
        self.city = city
        self.year = year
        self.month = month


# Create iterable page objects
# Ensure CSV directory exists
pages = []
for c in cities:
    for y in years:
        path = f"data/{c}/raw_csv/{y}"
        os.makedirs(path, exist_ok=True)
        for m in months:
            pages.append(page(c, y, m))

for p in pages:
    url = f"data/{p.city}/json/{p.year}/{p.month}.json"
    if not os.path.exists(url):
        continue

    with open(url, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["observations"])

    # Convert Unix timestamp to datetime
    df["datetime"] = pd.to_datetime(df["valid_time_gmt"], unit="s")

    # Keep only desired columns
    df = df[cols]

    # --- Numeric processing ---
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # Interpolate missing values
    interp = df[numeric_cols].interpolate(method='linear')
    df[numeric_cols] = interp.ffill().bfill()
    # Round numeric columns to integers where meaningful
    df[numeric_cols] = df[numeric_cols].round().astype(int)

    # --- Categorical processing ---
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")  # replace nulls

    # make non numeric columns numeric
    df = csvs.map_categorical_to_index(df)
    df = csvs.process_time_features(df)
    df = csvs.collapse_by_hour(df)

    # Save CSV
    csv_url = f"data/{p.city}/raw_csv/{p.year}/{p.month}.csv"
    df.to_csv(csv_url, index=False)
    print(f"DONE: {p.city}:{p.year}:{p.month}")


all_cities = {}
for city in cities:
    read_url = f"data/{city}/raw_csv"
    if not os.path.exists(read_url):
        continue

    print(f"Aggregating: {city}...")
    combined = []
    for y in years:
        y_url = os.path.join(read_url, str(y))
        if not os.path.exists(y_url):
            continue

        for m in months:
            m_url = os.path.join(y_url, f"{m}.csv")
            if not os.path.exists(m_url):
                continue
            df = pd.read_csv(m_url)
            combined.append(df)

    if len(combined) == 0:
        continue

    all_cities[city] = pd.concat(combined, ignore_index=True)

# Save all cities
all_cities = csvs.delete_unaligned_dates(all_cities)
for city, df in all_cities.items():
    df.to_csv(f"data/{city}.csv", index=False)

# Example usage for your city files (run before building df_list)
# for city in cities:
#     p = f"data/{city}.csv"
#     df = pd.read_csv(p)
#     df = trendy.add_trend_features(df, target_col="temp")
#     df.to_csv(p, index=False)   # overwrite (or save to new path)


for city in cities:
    for y in years:
        for m in months:
            # Delete raw_csvs
            csv_url = f"data/{city}/raw_csv/{y}"
            if not os.path.exists(csv_url + f"/{m}.csv"):
                continue
            os.remove(csv_url + f"/{m}.csv")

            # # Delete json
            # json_url = f"data/{city}/json/{y}"
            # if not os.path.exists(csv_url + f"/{m}.json"):
            #     continue
            # os.remove(csv_url + f"/{m}.json")

        # Delete aggregate year csvs
        os.removedirs(f"data/{city}/raw_csv/{y}")
        if not os.path.exists(f"data/{city}/{y}.csv"):
            continue
        os.remove(f"data/{city}/{y}.csv")

    # Remove City Dir
    # os.removedirs(f"data/{city}")
