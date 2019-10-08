# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:13:37 2019

@author: adamm
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import tensorflow as tf
from sklearn.utils import shuffle
import imageio

figsize = (12, 6)
np.random.seed(1)


def create_dataset_dataframe(filename):
    originaldata = pd.read_csv(filename)
    dataset = pd.DataFrame({'date': [i for i in range(len(originaldata['date']))], 'price': originaldata['close']})
    return dataset


def split_train_test(dataset, alpha=0.7):
    train = dataset[:int(alpha*len(dataset))]
    test = dataset[int(alpha*len(dataset)):]
    return train, test


def create_gif():
    images = []
    os.remove('Plots/training_evolution.gif')
    for i, _ in enumerate(os.listdir('Plots')):
        images += [imageio.imread('Plots/img_e%d.png' % i)]
        os.remove('Plots/img_e%d.png' % i)
    imageio.mimsave('Plots/training_evolution.gif', images, duration=0.5)


def upgrade_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['price'].rolling(window=7).mean()
    dataset['ma21'] = dataset['price'].rolling(window=21).mean()

    # Create MACD
    dataset['26ema'] = dataset['price'].ewm(span=26).mean()
    dataset['12ema'] = dataset['price'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])
    # Create Bollinger Bands
    dataset['20sd'] = dataset['price'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)

    # Create Exponential moving average
    dataset['ema'] = dataset['price'].ewm(com=0.5).mean()

    # Create Momentum
    dataset['momentum'] = dataset['price']-1
    return dataset


def plot_technical_indicators(train, test, indicators=['price']):
    plt.figure(figsize=figsize)
    for ind in indicators:
        plt.plot(train['date'], train[ind], label=ind+" train")
        plt.plot(test['date'], test[ind], label=ind+" test")
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Evolution of Goldman Sachs technical indicators')
    plt.legend()
    plt.show()


def create_set(train, test, size, ind="all"):
    if ind == "all":
        ind = [i for i in train.columns if i != 'date']
    X_train = np.zeros((len(train)-size-1, 100, len(ind)))
    y_train = np.zeros((len(train)-size-1, len(ind)))
    X_test = np.zeros((len(test)-1, 100, len(ind)))
    y_test = np.zeros((len(test)-1, len(ind)))
    for i in range(len(train)-size-1):
        X_train[i] = train[ind].iloc[i:i+size]
        y_train[i] = train[ind].iloc[i+size]
    for i in range(len(test)-1):
        if i < size:
            X_test[i] = np.concatenate((train[ind].iloc[-size+i:], test[ind].iloc[:i]))
            y_test[i] = test[ind].iloc[i]
        else:
            X_test[i] = test[ind].iloc[i-size:i]
            y_test[i] = test[ind].iloc[i]
    return X_train.astype('float32'), y_train.astype('float32'), X_test.astype('float32'), y_test.astype('float32')


class TCN():
    def __init__(self, nb_features, input_size=100):
        super().__init__()
        if 'mod.h5' in os.listdir('Models'):
            self.model = tf.keras.models.load_model('Models/mod.h5')
            print("Model loaded!")
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(input_size, nb_features), dtype=tf.float32),
                tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
                tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100, activation='relu'),
                tf.keras.layers.Dense(nb_features)
            ])
        self.input_size = input_size
        self.optimizer = tf.keras.optimizers.Adam()

    def compute_loss(self, y_true, y_pred):
        return tf.losses.MSE(y_true, y_pred)

    def train_step(self, X_batch, y_batch):
        with tf.GradientTape() as t:
            out = self.model(X_batch)
            loss = self.compute_loss(y_batch, out)
        gradients = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def train(self, X_train, y_train, X_test, y_test, epochs=1000, batch_size=32):
        print("Start training!")
        init_y_train = y_train
        batch_number = int(len(y_train)//batch_size)
        for epoch in range(epochs):
            (X_train, y_train) = shuffle(X_train, y_train)
            for batch in range(batch_number):
                X_batch = X_train[batch*batch_size:(batch+1)*batch_size]
                y_batch = y_train[batch*batch_size:(batch+1)*batch_size]
                self.train_step(X_batch, y_batch)
            test_loss = self.compute_loss(self.model(X_test), y_test)
            print("Epoch %d | loss = %.3f" % (epoch+1, tf.reduce_mean(test_loss)))
            if (epoch+1) % 1 == 0:
                self.test(init_y_train, X_test, y_test, save=True, epoch=epoch)
        self.model.save('Models/mod.h5')
        print("Model saved!")
        create_gif()

    def test(self, y_train, X_test, y_test, save=True, epoch=1):
        plt.figure(figsize=figsize)
        plt.plot(y_train[:, 0])
        plt.plot([i for i in range(len(y_train), len(y_train)+len(y_test))], y_test[:, 0])
        y_pred = self.model(X_test)
        plt.plot([i for i in range(len(y_train), len(y_train)+len(y_test))], y_pred[:, 0])
        plt.title("Analysis of the prediction at epoch %d" % epoch)
        plt.xlabel("Date")
        plt.ylabel("Price")
        if save:
            plt.savefig("Plots/img_e%d" % epoch)
        else:
            plt.show()


if __name__ == '__main__':
    dataset = create_dataset_dataframe("GS.csv")
    dataset = upgrade_technical_indicators(dataset)
    train, test = split_train_test(dataset)
    ind = ["price"]
    # plot_technical_indicators(train, test, ['price', 'ma7', 'ma21'])
    X_train, y_train, X_test, y_test = create_set(train, test, 100, ind)
    tcn = TCN(len(ind))
    tcn.train(X_train, y_train, X_test, y_test, 20)
