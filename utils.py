import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
# utils
def plot_multi_dataframe(df, startIndex, endIndex):
    plt.figure(figsize=(15,15))
    column_nr = df.shape[1]
    for i in range(0, column_nr):
        plt.subplot(column_nr, 1, i+1)
        sub_df = df[[df.columns[i]]]
        plt.plot(sub_df)
        plt.title(df.columns[i], y=0.5, loc='right')    
    plt.show()

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


def plot_results(real, predicted, test):
    # plot
    column_nr = real.shape[1]
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
        plt.title(test.columns[i] + rmse + mae)
    plt.show()