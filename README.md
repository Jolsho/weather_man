## TRAINING_OUTPUT::
    
    Epoch 01 | Train Loss: 0.3584 | Val Loss: 0.2103
    Epoch 02 | Train Loss: 0.1615 | Val Loss: 0.1823
    Epoch 03 | Train Loss: 0.1317 | Val Loss: 0.1562
    Epoch 04 | Train Loss: 0.0956 | Val Loss: 0.1061
    Epoch 05 | Train Loss: 0.0780 | Val Loss: 0.1134
    Epoch 06 | Train Loss: 0.0705 | Val Loss: 0.0925
    Epoch 07 | Train Loss: 0.0669 | Val Loss: 0.0922
    Epoch 08 | Train Loss: 0.0644 | Val Loss: 0.1020
    Epoch 09 | Train Loss: 0.0631 | Val Loss: 0.0966
    Epoch 10 | Train Loss: 0.0602 | Val Loss: 0.0928

    Validation performance (model):
    Horizon t+1: RMSE=2.64, MAE=2.11, R²=0.932
    Horizon t+3: RMSE=2.89, MAE=2.29, R²=0.919
    Horizon t+6: RMSE=3.25, MAE=2.57, R²=0.897
    Horizon t+12: RMSE=3.40, MAE=2.65, R²=0.888
    Horizon t+24: RMSE=3.76, MAE=2.95, R²=0.863

    Validation performance (baseline):
    [Baseline] Horizon t+1: RMSE=3.77, MAE=2.76, R²=0.872
    [Baseline] Horizon t+3: RMSE=3.77, MAE=2.76, R²=0.872
    [Baseline] Horizon t+6: RMSE=3.77, MAE=2.77, R²=0.872
    [Baseline] Horizon t+12: RMSE=3.77, MAE=2.77, R²=0.872
    [Baseline] Horizon t+24: RMSE=3.77, MAE=2.76, R²=0.872

    Test performance (model):
    Horizon t+1: RMSE=2.87, MAE=2.21, R²=0.920
    Horizon t+3: RMSE=3.15, MAE=2.40, R²=0.904
    Horizon t+6: RMSE=3.50, MAE=2.66, R²=0.881
    Horizon t+12: RMSE=3.59, MAE=2.76, R²=0.875
    Horizon t+24: RMSE=3.63, MAE=2.82, R²=0.872

    Test performance (baseline):
    [Baseline] Horizon t+1: RMSE=3.77, MAE=2.76, R²=0.872
    [Baseline] Horizon t+3: RMSE=3.77, MAE=2.76, R²=0.872
    [Baseline] Horizon t+6: RMSE=3.77, MAE=2.77, R²=0.872
    [Baseline] Horizon t+12: RMSE=3.77, MAE=2.77, R²=0.872
    [Baseline] Horizon t+24: RMSE=3.77, MAE=2.76, R²=0.872

    Sample prediction (°C):
    Predicted (t+1, t+3, t+6, t+12, t+24): [33.98669134 33.67518658 34.49643588 38.27158712 34.40594391]
    True      (t+1, t+3, t+6, t+12, t+24): [32.00000022 30.99999975 33.9999999  40.99999941 32.99999943]


## SUMMARY::
### MODEL:: 
LSTM was chosen because of the time-series nature of this. Needed a seperate hidden layer and output for each horizon. Architecturally simple, I personally didn't have to do a ton of work. Because PyTorch does so much for me under the hood. I could try different architectures, but am just not that interested.
Started this project with consistent MAE=4 and R2=0.8 from t+1 -> t+6. I foolishly thought I wouldnt include the target locations current data. Quite obviously that was the largest increase in accuracy. The second largest factor was the hidden layers size. Started at 128, and stepped to 64, and settled at 32. I tried to derive some trend metrics to each row, but this suprisingly didnt do much. Learning rate was also initially 1e-4 but was overshooting too much. Tried to increase sequence length, to get say the last 24 hours of data. Turns out that didn't have a massive effect except in training time.
Las Vegas has quite predictable weather, except for the occasional abnormal pattern. Given such its obvious why the baseline was already so high. The baseline I chose was just simply the previous days temp at that hour. I could slightly outperform it in the near term, which means this could capture some local dynamics.

### DATA::
This was trained on three years of hourly averages. There are roughly 15 features, excluding derived ones. If I were to do this more thoroughly I would say I likely need more data. Both in terms of more sophisticated trend metrics, as well as more locations. I don't think three years is a bad size although it wouldn't hurt to get more.
All the data was sourced from a wunderground.com. I essentially had to make a request in my browser, find the request in the network tab of dev-tools and copy the API key. Then I created an automated script which would use that api key and a predefined set of locations and years to fetch the data. At the end of this there are just a ton of json files, so I had to put those into dataframes, and clean them up. I was able to automate away missing entries, missing values in provided entries, and missalignments in time between locations. In the end of the json_to_csv.py script there are len(cities) files with hourly metrics for a total of 3 years.
In all this weather_man is not very good, and is no where near useful accuracy. At the same time however it was built in 2 weeks with an a minimal dataset. Was a good learning experience though.
