import requests
import json
import os
from datetime import datetime, timedelta
import calendar
from structs import cities, stations


# Function to generate start/end dates for each month
def generate_month_date_ranges(start_year, end_year):
    date_ranges = []
    current = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)

    while current <= end:
        start_date = current.strftime("%Y%m%d")
        next_month = current.replace(day=28) + timedelta(days=4)
        last_day = next_month - timedelta(days=next_month.day)
        end_date = last_day.strftime("%Y%m%d")

        year = current.year
        month_name = calendar.month_name[current.month]

        date_ranges.append((year, month_name, start_date, end_date))
        current = last_day + timedelta(days=1)

    return date_ranges


# Generate date ranges for each month
month_ranges = generate_month_date_ranges(2019, 2021)

"""
0: Kingman
1: Henderson
2: Nellis
3: Vegas
4: IndianSprings
"""
offset = 0
ending = 5
api_key = "NEED TO FILL THIS IN TO USE THIS SCRIPT"

for i, s in enumerate(stations[offset:ending]):
    url = f"https://api.weather.com/v1/location/{s}{
                            ":9:US/observations/historical.json"}"
    city = cities[i+offset]

    print(f"\n Fetching data for {city}...")
    current_year = ""
    # Fetch data and organize into folders by year/month
    for year, month_name, start_date, end_date in month_ranges:
        params = {
            "apiKey": api_key,
            "units": "m",
            "startDate": start_date,
            "endDate": end_date
        }

        response = requests.get(url, params=params)

        if year != current_year:
            print(f"{year}, ", end="")
            current_year = year

        if response.status_code == 200:
            data = response.json()

            # Create folder for the year if it doesn't exist
            year_folder = f"data/{city}/json/{year}"
            os.makedirs(year_folder, exist_ok=True)

            # Save file as month.json
            file_path = f"{year_folder}/{month_name}.json"
            with open(file_path, "w") as file:
                json.dump(data, file, indent=2)
        else:
            print(f"Request failed for {city}{year}{month_name}: {response.status_code}")
