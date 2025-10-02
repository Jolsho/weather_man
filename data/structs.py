stations = [
    "KIFP", "KBVU",
    "KLSV", "KLAS",
    "KVGT"
]

cities = [
    "Kingman", "Henderson",
    "Nellis", "Vegas",
    "IndianSprings"
]

months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August", "September",
    "October", "November", "December",
]

years = ["2019", "2020", "2021"]

# Columns to include
cols = [
    "datetime", "temp", "dewPt", "rh", "pressure",
    "wdir", "wspd", "uv_index",
    "vis", "clds", "wdir_cardinal", "heat_index",
    "wc", "feels_like", "precip_hrly"
]

# Numeric columns for interpolation
numeric_cols = [
    "temp", "dewPt", "rh",
    "pressure", "wdir", "wspd",
    "uv_index", "vis",
    "heat_index", "wc",
    "feels_like", "precip_hrly"
]

# Categorical columns
categorical_cols = ["clds", "wdir_cardinal"]

# i = -5 = wdir-card
windDirs = [
    "N", "NNW", "NW", "WNW",
    "W", "WSW", "SW", "SSW",
    "S", "SSE", "SE", "ESE",
    "E", "ENE", "NE", "NNE",
    "CALM", "Unknown", "VAR",
]

# i = -6 = clds
clds = ["OVC", "BKN", "CLR", "SCT", "FEW", "Unknown"]

feature_cols = [
    "month", 'hour_sin', 'hour_cos'
    "temp", "temp_avg3", "dewPt",
    "rh", "pressure",
    "wdit", "wspd", "uv_index",
    "vis", "clds", "wdir_cardinal",
    "heat_index", "wc", "feels_like",
    "precip_hrly",
]
