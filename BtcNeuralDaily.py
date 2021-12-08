import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
from SetNeuralNetwork import set_neural_network, set_x_train, set_y_train, set_x_test, set_y_test, \
    set_data_test, set_data_train, visualize_training_results, set_y_train_linear, set_y_test_linear, \
    learning_rate, epochs, batch_size
from buildtable import build_table
#### GET DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data = pd.read_csv("csv/prepared_data.csv")
#data = data[12:70] ####To remove~- minimize sample to be faster coding
#### Normalization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
#### SPLIT DATA between traing and testing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data_train = set_data_train(data)
data_test = set_data_test(data)
#### Define X_train (data) & Y_train (labels) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### Define X_test (data) & Y_test (labels) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
X_train = set_x_train(data_train)
Y_train = set_y_train(data_train)
X_test = set_x_test(data_test)
Y_test = set_y_test(data_test)
#### Build neural network ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
learning_rate = learning_rate()
epochs = epochs()
batch_size = batch_size()
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 monitor = "val_mean_absolute_error",
                                                 #monitor = "loss",
                                                 save_best_only = True,
                                                 save_freq = "epoch")

neural_network = set_neural_network(X_train, Y_train)

neural_network.compile(loss = tf.losses.MeanSquaredError(),
                       optimizer = tf.optimizers.Adam(lr=learning_rate, amsgrad=True),
                       metrics=['mean_absolute_error', 'accuracy']
                       )

neural_network.summary()
latest = tf.train.latest_checkpoint(checkpoint_dir)
#neural_network.load_weights(latest) # hide to start over

results = neural_network.fit(X_train, Y_train, batch_size = batch_size, epochs=epochs, validation_data=(X_test, Y_test),  callbacks=[cp_callback], shuffle=True)

visualize_training_results(results, 'loss')
visualize_training_results(results, 'val_mean_absolute_error')

plt.show

from do_prediction_btcdaily import do_prediction
