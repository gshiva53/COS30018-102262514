# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
# import matplotlib.pyplot as plt
import mplfinance as fplt
# from bqplot import pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from pandas_datareader import data
from sklearn.utils.validation import column_or_1d
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import pdb

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

# Global Values
TRAIN_START = dt.datetime(2012, 5, 23)  # Start date to read
TRAIN_END = dt.datetime(2020, 1, 7)  # End date to read
DATA_SOURCE = "yahoo"
COMPANY = "FB"
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Volume', 'Close', 'Adj Close']
PRICE_VALUE = FEATURE_COLUMNS[4]
PREDICTION_DAYS = 60  # Original
scaler = MinMaxScaler(feature_range=(0, 1))
x_train, y_train = [], []
x_test, y_test = [], []
column_scaler = {}
original_data = []
data = []


def shuffle_in_unison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


# create these folders if they do not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")


def load_data(ticker=COMPANY, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
              test_size=0.2, feature_columns=FEATURE_COLUMNS,
              prediction_days=PREDICTION_DAYS, start_date=TRAIN_START, end_date=TRAIN_END):
    global x_train, y_train
    global x_test, y_test
    global data
    global scaled_data
    global column_scaler
    global scaler
    global original_data
    global PRICE_VALUE
    global DATA_SOURCE
    global PREDICTION_DAYS

    if isinstance(ticker, str):
        data = web.DataReader(ticker, DATA_SOURCE, start_date, end_date)
    elif isinstance(ticker, pd.DataFrame):
        data = ticker
    else:
        raise TypeError("ticker can be either a str or a 'pd.DataFrame' instance")

    for col in feature_columns:
        assert col in data.columns, f"'{col}' does not exist in the dataframe."

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1)) 

    if scale:
        for column in feature_columns:
            # PRICE_VALUE = column
            # data[column] = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))
            scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))
            column_scaler = scaler

    data.dropna(inplace=True)

    # return the original dataset 
    original_data = data.copy()
    # pdb.set_trace()
    scaled_data = scaled_data[:, 0]  # Turn the 2D array back to a 1D array
    # pdb.set_trace()
    # Prepare the data
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x])
        y_train.append(scaled_data[x])

    # Convert them into an array
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
    # and q = PREDICTION_DAYS; while y_train is a 1D array(p)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
    # is an array of p inputs with each input being a 2D array 

    # result = {}
    # if "date" not in data.columns: 
    #     data["date"] = data.index

    if split_by_date:
        train_samples = int((1 - test_size) * len(x_train))
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]

        x_test = x_train[train_samples:]
        y_test = y_train[train_samples:]
        if shuffle:
            shuffle_in_unison(x_train, y_train)
            shuffle_in_unison(x_test, y_test)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, shuffle=shuffle)


load_data()

model = Sequential()  # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.

# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=5, batch_size=32)

TEST_START = dt.datetime(2020, 1, 8)
TEST_END = dt.datetime.now()

test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)
# Again, I saw a bug with pandas_datareader.DataReader() as it
# includes the date 07/01/2020 in test_data!
# We'll call this ISSUE #1

# The above bug is the reason for the following line of code
test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values
# pdb.set_trace()
total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)

# ------------------------------------------------------------------------------
# Make predictions on test data
# ------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
# ------------------------------------------------------------------------------
# Plot the test predictions
# ------------------------------------------------------------------------------

# plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
# plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
# plt.title(f"{COMPANY} Share Price")
# plt.xlabel("Time")
# plt.ylabel(f"{COMPANY} Share Price")
# plt.legend()
# plt.show()

# fig = plt.figure(f"{COMPANY} Share Price")
# fig.layout.width="800px"
# ohlc = plt.ohlc(x=actual_prices, y=actual_prices, marker='candle')
# ohlc.colors=["lime", "tomato"]
# plt.xlabel("Time")
# plt.ylabel(f"{COMPANY} Share Price")
# plt.show()


fplt.plot(
    type='candle',
    ylabel=f"Actual {COMPANY} Price"
)
fplt.title(f"{COMPANY} Share Price")
fplt.xlabel("Time")
fplt.ylabel(f"{COMPANY} Share Price")
fplt.legend()
fplt.show()

# ------------------------------------------------------------------------------
# Predict next day
# ------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??
