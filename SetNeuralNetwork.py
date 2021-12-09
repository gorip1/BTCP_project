import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
import pandas as pd

#### NN parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def training_prct():
    training_prct = 0.75
    return training_prct

def n_days_used_to_predict():
    n_days_used_to_predict = 2
    return n_days_used_to_predict

def n_days_in_the_future():
    n_days_in_the_future= 0
    return n_days_in_the_future

def set_neural_network(X_train, Y_train):
    neural_network = tf.keras.Sequential([
        layers.LSTM(units = 128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'),
        layers.Dropout(0.3),
        layers.LSTM(units=64, return_sequences=False, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])
    return neural_network

def learning_rate():
    learing_rate = 0.001
    return learing_rate

def batch_size():
    batch_size = 25
    return batch_size

def epochs():
    epochs = 300
    return epochs

### set training/test data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def set_x_train(data_train):
    x_train = []
    for i in range(n_days_used_to_predict(), data_train.shape[0]-n_days_in_the_future()):
            x_train.append(data_train[i-n_days_used_to_predict():i])
    x_train = np.array(x_train)
    return x_train

def set_y_train(data_train):
    y_train = []
    for i in range(n_days_used_to_predict(), data_train.shape[0]-n_days_in_the_future()):
            y_train.append(data_train[i+n_days_in_the_future(),0])
    y_train = np.array(y_train)
    return y_train

def set_x_test(data_test):
    x_test = []
    for i in range(n_days_used_to_predict(), data_test.shape[0]-n_days_in_the_future()):
            x_test.append(data_test[i-n_days_used_to_predict():i])
    x_test = np.array(x_test)
    return x_test

def set_y_test(data_test):
    y_test = []
    for i in range(n_days_used_to_predict(), data_test.shape[0]-n_days_in_the_future()):
            y_test.append(data_test[i+n_days_in_the_future(),0])
    y_test = np.array(y_test)
    return y_test

def set_data_train(data):
    data_train = data[0:int(training_prct()*len(data))] #### 75% training
    return data_train

def set_data_test(data):
    data_test = data[int(training_prct()*len(data)):len(data)] #### 25% testing
    return data_test

### Visualization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def visualize_training_results(results, parameter):
    history = results.history
    plt.figure(figsize=(12, 4))
    plt.plot(history[parameter])
    plt.title(parameter)
    plt.xlabel('Epochs')
    plt.ylabel(parameter)


### test prediction with linear regression of n_days

def set_y_train_linear(data_train):
    y_train = []
    for i in range(n_days_used_to_predict(), data_train.shape[0]-n_days_in_the_future()):
        data_ = []
        for j in range(0, n_days_in_the_future()):
            data_.append(data_train[i+j, 1])
        df = pd.DataFrame(data_)
        df.reset_index(drop=False, inplace=True)
        X = df['index'].values.reshape(-1, 1)
        Y = df[0].values.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(X, Y)
        coef = reg.coef_[0][0]
        y_train.append(coef)
    y_train = np.array(y_train)
    return y_train

def set_y_test_linear(data_test):
    y_test = []
    for i in range(n_days_used_to_predict(), data_test.shape[0] - n_days_in_the_future()):
        data_ = []
        for j in range(0, n_days_in_the_future()):
            data_.append(data_test[i + j, 1])
        df = pd.DataFrame(data_)
        df.reset_index(drop=False, inplace=True)
        X = df['index'].values.reshape(-1, 1)
        Y = df[0].values.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(X, Y)
        coef = reg.coef_[0][0]
        y_test.append(coef)
    y_test = np.array(y_test)
    return y_test
