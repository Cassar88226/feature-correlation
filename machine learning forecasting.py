import pandas as pd
import numpy as np
import xlrd
from matplotlib import pyplot as plt
from utils import *

#read the data
df = pd.read_excel("Data Inputs.xlsx", skiprows=1)

# drop unwanted columns
data = df.drop(['US', 'SeriesDate'], axis=1)
data.index = df.SeriesDate

# plot data
plot_multi_dataframe(data, 0, len(data))

# define parameters
n_forecast_days = 30
n_lag_days = 7
window_size = 2
n_trees = 100

# Feature Engineering
# generate 2 window rolling feature
means = rolling_feature(data, window_size)
# genearate 14 Lag Features
lagged_data = lagged_feature(data, n_lag_days)

# merge lagged and rolling mean features
dataframe = pd.concat([lagged_data, means], axis=1)
# drop nan rows
dataframe = dataframe.dropna()
print(dataframe.head())


y_label = data.columns
test_valid = dataframe[len(dataframe)-n_forecast_days:]
y_actual = test_valid[y_label].values



# get models list
models = get_models(n_trees)

# evaluate the models
for name, model in models.items():
    evaluate_model(name, model, n_forecast_days, dataframe, y_actual, test_valid, y_label)
