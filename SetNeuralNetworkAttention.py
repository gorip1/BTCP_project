import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, Input, Dense, SimpleRNN, LSTM, Dropout
import tensorflow.keras.backend as K


def training_prct():
    training_prct = 0.75
    return training_prct

def n_days_used_to_predict():
    n_days_used_to_predict = 14
    return n_days_used_to_predict

def n_days_in_the_future():
    n_days_in_the_future= 1
    return n_days_in_the_future

def learning_rate():
    learing_rate = 0.001
    return learing_rate

def batch_size():
    batch_size = 25
    return batch_size

def epochs():
    epochs = 50
    return epochs


    # Add attention layer to the deep learning network
    class attention(Layer):
        def __init__(self, **kwargs):
            super(attention, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                     initializer='random_normal', trainable=True)
            self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                     initializer='zeros', trainable=True)
            super(attention, self).build(input_shape)

        def call(self, x):
            # Alignment scores. Pass them through tanh function
            e = K.tanh(K.dot(x, self.W) + self.b)
            # Remove dimension of size 1
            e = K.squeeze(e, axis=-1)
            # Compute the weights
            alpha = K.softmax(e)
            # Reshape to tensorFlow format
            alpha = K.expand_dims(alpha, axis=-1)
            # Compute the context vector
            context = x * alpha
            context = K.sum(context, axis=1)
            return context


    def create_RNN_with_attention():
        print(X_train.shape[1], X_train.shape[2])
        x = Input(shape=(X_train.shape[1], X_train.shape[2]))
        LSTM1 = LSTM(128, return_sequences=True, activation='relu')(x)
        Drop = Dropout(0.5)(LSTM1)
        LSTM2= LSTM(32, return_sequences=True, activation='relu')(Drop)
        Drop2 = Dropout(0.3)(LSTM2)
        attention_layer = attention()(Drop2)
        outputs = Dense(1)(attention_layer)
        model = Model(x, outputs)
        return model


    # Create the model with attention, train and evaluate
    neural_network = create_RNN_with_attention()
    neural_network.summary()
    return neural_network
