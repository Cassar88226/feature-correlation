import pandas as pd
import numpy as np
import xlrd
from matplotlib import pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from utils import *

#read the data
df = pd.read_excel("Data Inputs.xlsx", skiprows=1)
df.head()

# drop unwanted columns
data = df.drop(['US', 'SeriesDate'], axis=1)
data.index = df.SeriesDate
data.head()

# plot data
plot_multi_dataframe(data, 0, len(data))

# define parameters
n_forecast_days = 30
n_lag_days = 30

#creating the train and test set
#creating the train and test set
values = data.values
train = data[:len(data)-n_forecast_days]
test = data[len(data)-n_forecast_days:]
print(train.head())
print(test.head())


# plot train and test data
plot_train_test_dataframe(train, test)

#fit the model and make prediction on test
history = train.copy(deep=True)
predictions = list()
for time in range(len(test)):
    model = VAR(train)
    model_fit = model.fit(n_lag_days)
#     yhat = model_fit.forecast(history.values, 1)[0][0]
    yhat = model_fit.forecast(history.values, 1)[0]
    predictions.append(yhat.tolist())
    history = history.append(test[time:time+1])
# y_actual = test.values[:,0]
y_actual = test.values
predictions = np.asarray(predictions)

plot_results(y_actual, predictions, test)