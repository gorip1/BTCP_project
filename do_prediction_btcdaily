import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 100)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
from SetNeuralNetwork import set_neural_network, set_x_train, set_y_train, set_x_test, set_y_test, set_data_test, set_data_train, visualize_training_results, set_y_train_linear, set_y_test_linear

def do_prediction():

    #### GET DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data = pd.read_csv("csv/prepared_data.csv")
    #data = data[12:70] ####To remove~- minimize sample to be faster coding
    #### Normalization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    #### SPLIT DATA between traing and testing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data_train = set_data_train(data)
    data_test = set_data_test(data)
    #### Define X_train (data) & Y_train (labels) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #### Define X_test (data) & Y_test (labels) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X_train = set_x_train(data_train)
    Y_train = set_y_train(data_train)
    X_test = set_x_test(data_test)
    Y_test = set_y_test(data_test)
    #### Build neural network ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    checkpoint_path = "checkpoint/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    neural_network = set_neural_network(X_train, Y_train)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    neural_network.load_weights(latest).expect_partial()
    #### run pred ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    prediction = neural_network(X_test)

    prediction = prediction.numpy()
    pred_simp = []
    X_test_last_value = []

    for i in prediction:
      pred_simp.append(i[0])
    comparator = pd.DataFrame()
    comparator['réalité'] = Y_test
    comparator['prediction'] = prediction

    Y_test_dollard = []
    for i in range(0,len(Y_test)):
        line_t = [Y_test[i]]
        for j in range(0, data.shape[1]-1):
            line_t.append(0)
        Y_test_dollard.append(line_t)
    Y_test_dollard = scaler.inverse_transform(Y_test_dollard)
    Y_test_dollard = pd.DataFrame(Y_test_dollard)

    Y_pred_dollard = []
    for i in range(0,len(prediction)):
        line_p = [prediction[i,0]]
        for j in range(0, data.shape[1] - 1):
            line_p.append(0)
        Y_pred_dollard.append(line_p)

    Y_pred_dollard = scaler.inverse_transform(Y_pred_dollard)
    Y_pred_dollard = pd.DataFrame(Y_pred_dollard)

    final_results = pd.DataFrame()
    final_results['Réalité'] = Y_test_dollard[0]
    final_results['Prédiction'] = Y_pred_dollard[0]

    indexnames = final_results[final_results['Prédiction']*final_results['Réalité'] < 0 ].index
    Accuracy = len(indexnames)/len(final_results)
    print(final_results)
    print("up and down accuracy = ", round(Accuracy,2))

    final_results.plot(figsize=(15,6), xticks=range(0, 8), title='BTC vs PRED, updown accuracy = '+ str(round(Accuracy,2))).legend(title='BTC vs PRED', bbox_to_anchor=(1, 1))
    plt.show()
    return()

do_prediction()
