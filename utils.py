import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# ensemble method
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.ensemble import GradientBoostingRegressor

# non-linear method
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

# linear models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
# from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoLars
# from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor


def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_error(ytrue, ypred))
def rmse(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))
def mae():
    return np.sqrt(mean_absolute_error(ytrue, ypred))

# plot Multiple Time Series Dataframe
def plot_multi_dataframe(df, startIndex, endIndex):
    plt.figure(figsize=(15,15))
    column_nr = df.shape[1]
    for i in range(0, column_nr):
        plt.subplot(column_nr, 1, i+1)
        sub_df = df[[df.columns[i]]]
        plt.plot(sub_df)
        plt.title(df.columns[i], y=0.5, loc='right')    
    plt.show()

# plot Train Test Dataframe
def plot_train_test_dataframe(train, test):
    plt.figure(figsize=(15,15))
    column_nr = train.shape[1]
    for i in range(0, column_nr):
        plt.subplot(column_nr, 1, i+1)
        sub_train = train[[train.columns[i]]]
        sub_test = test[[test.columns[i]]]
        plt.plot(sub_train)
        plt.plot(sub_test)
        plt.title(train.columns[i], y=0.5, loc='right')    
    plt.show()

# plot real and prediction
def plot_results(model_name, real, predicted, test):
    # plot
    column_nr = real.shape[1]
    print(model_name)
    plt.figure(figsize=(15,20))
    for i in range(column_nr):
        plt.subplot(column_nr, 1, i+1)
        rmse = np.sqrt(mean_squared_error(real[:, i], predicted[:, i]))
        mae = mean_absolute_error(real[:, i], predicted[:, i])
        rmse = ' RMSE: %.3f' % rmse
        mae = ' MAE: %.3f' % mae
        real_df = pd.DataFrame(index=test.index,columns=[test.columns[i]], data=real[:, i])
        predicted_df = pd.DataFrame(index=test.index,columns=[test.columns[i]], data=predicted[:, i])
        plt.plot(real_df)
        plt.plot(predicted_df, color='red')
        plt.legend(['real','prediction'],loc='upper left')
        plt.subplots_adjust(hspace = 0.5)
        # plt.legend(loc="upper right")
        title = test.columns[i] + rmse + mae
        plt.title(title)
        print(title)
    # plt.show()
    plt.savefig(model_name +'.png')
#read the data
def read_data(file, skiprows):
    df = pd.read_excel(file, skiprows=skiprows)
    return df

# plot Autocorrelation
def plot_acf(data, column, lags):
    plot_acf(data[column], lags=lags)

# plot Partial Autocorrelation
def plot_pacf(data, column, lags):
    plot_pacf(data[column], lags=lags)

# Rolling Feature
def rolling_feature(data, window_size):
    # rolling window size should be greater 1
    if window_size < 2 :
        window_size = 2
    temps = pd.DataFrame(data.values, index = data.index)
    # shift data
    shifted = temps.shift(1)
    # rolling mean
    window = shifted.rolling(window=window_size)
    means = window.mean()
    # generate columns
    means_columns = []
    for column in data.columns:
        means_columns.append(column + ' mean(t-' + str(window_size) + ', t-1)')
    means.columns = means_columns
    return means

# Lag Feature
def lagged_feature(data, lags):
    lagged_data = data.copy()
    for i in range(1, lags+1):
        for col in data.columns:
            lagged_data[col + " lag_" + str(i)] = data[col].shift(i)
    return lagged_data

# get models dict
def get_models(n_trees = 100):
    models=dict()
    models['LinearRegression'] = LinearRegression()
    models['Lasso'] = Lasso()
    models['Ridge'] = Ridge()
    models['ElasticNet'] = ElasticNet()
    models['LassoLars'] = LassoLars()
    models['BaggingRegressor'] = BaggingRegressor(n_estimators=n_trees, n_jobs=-1, random_state=0)
    models['RandomForestRegressor'] = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=0)
    models['ExtraTreesRegressor'] = ExtraTreesRegressor(n_estimators=n_trees, n_jobs=-1, random_state=0)
    # models['GradientBoostingRegressor'] = GradientBoostingRegressor(n_estimators=n_trees, random_state=0)

    models['KNeighborsRegressor'] = KNeighborsRegressor(n_neighbors=7)
    models['DecisionTreeRegressor'] = DecisionTreeRegressor(random_state=0)
    models['ExtraTreeRegressor'] = ExtraTreeRegressor(random_state=0)
    return models

# evaluate the model
def evaluate_model(name, model, n_forecast_days, dataframe, y_actual, test_valid, y_label):
    predictions = list()
    for time in range(n_forecast_days):
        train_end_idx = len(dataframe)-n_forecast_days + time
        train = dataframe[:train_end_idx]
        test = dataframe[train_end_idx:train_end_idx+1]
        xtr, xts = train.drop(y_label, axis=1), test.drop(y_label, axis=1)
        ytr, yts = train[y_label].values, test[y_label].values
        
        model.fit(xtr, ytr)
        yhat = model.predict(xts)[0]
        predictions.append(yhat.tolist())

    predictions = np.asarray(predictions)
    plot_results(name, y_actual, predictions, test_valid)