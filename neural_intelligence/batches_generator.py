import csv

import matplotlib.pyplot as plt
import numpy as np
from constants.constants import Constants
from indicators.rsi import RSI


def generate_batch_yielded(pair, sequence_length=60):
    with open("resources/" + pair + ".csv", mode='r') as csv_file:
        data = np.array(list(csv.reader(csv_file)))
        clean_data = data[:, [Constants.CANDLE_OPEN, Constants.CANDLE_HIGH, Constants.CANDLE_LOW,
                              Constants.CANDLE_CLOSE, Constants.QUOTE_VOLUME]].astype(float)
        for i in range(len(clean_data) - sequence_length - 1):
            print(clean_data[i:i + sequence_length])
            print(clean_data[i + sequence_length + 1][Constants.CANDLE_CLOSE])
            x_train = clean_data[i:i + sequence_length]
            y_train = clean_data[i + sequence_length + 1, Constants.CANDLE_CLOSE]
        yield np.array(x_train), np.array(y_train)


def generate_batch(dataset, univariate_index, n_future, sequence_length=60):
    x_data, y_data, y_data_supplied = [], [], []
    for i in range(len(dataset) - sequence_length - 1 - n_future):
        x_data.append(dataset[i:i + sequence_length])
        y_data.append(dataset[i + sequence_length: i + sequence_length + n_future, [univariate_index]])
    return np.array(x_data), np.array(y_data)


def generate_smart_lstm_batch(dataset_x, dataset_y, sequence_length=30):
    x_data, y_data = [], []
    for i in range(len(list(zip(dataset_x, dataset_y))) - sequence_length - 1):
        x_data.append(dataset_x[i:i + sequence_length])
        y_data.append(dataset_y[i + sequence_length])
    return np.array(x_data), np.array(y_data)


def get_price_only(self, pair):
    with open("resources/" + pair + ".csv", mode='r') as csv_file:
        data = np.array(list(csv.reader(csv_file)))
    clean_data = data[:, Constants.CANDLE_CLOSE].astype(float)
    return clean_data


def get_smart_lstm_data(pair, lookback):
    data = np.array(calculate_sell_index(pair, lookback))
    data = data[:, [Constants.CANDLE_CLOSE, 12, 13], ].astype(float)
    data = RSI.calculate_rsi(data, lookback)
    return data[lookback:]


def get_clean_data(pair):
    with open("resources/" + pair + ".csv", mode='r') as csv_file:
        data = np.array(list(csv.reader(csv_file)))
        clean_data = data[:, [Constants.CANDLE_CLOSE, Constants.QUOTE_VOLUME, Constants.TAKER_BUY_ASSET_VOLUME]].astype(
            float)
        return clean_data


def get_all_data(pair):
    with open("resources/" + pair + ".csv", mode='r') as csv_file:
        data = np.array(list(csv.reader(csv_file))).astype(float)
        return data


def calculate_local_top(pair, lookback, lookup):
    data = get_all_data(pair)
    """
    Calculation of a BUY_INDEX for each row
    Would be the confidence into buying or selling an index
    0 --> strong SELLING feeling 
    1 --> strong BUYING feeling
    """
    BUY_INDEX = np.zeros((len(data), 1))

    for index, row in enumerate(data):
        if index < lookback or index > len(data) - 60:
            continue

        max_price = np.amax(data[index - lookback:index + lookup][:, Constants.CANDLE_CLOSE])
        min_price = np.amin(data[index - lookback:index + lookup][:, Constants.CANDLE_CLOSE])
        BUY_INDEX[index][0] = (row[Constants.CANDLE_CLOSE] - min_price) / (max_price - min_price)

    data_with_buy_index = np.hstack((data, BUY_INDEX))[lookback: len(data) - lookup]
    indexed_data = np.hstack((data_with_buy_index, np.arange(len(data_with_buy_index)).reshape(-1, 1)))
    return indexed_data


def calculate_sell_index(pair, lookup):
    data = get_all_data(pair)

    """
     Calculation of a BUY_INDEX for each row
     Would be the confidence into buying or selling an index
     0 --> strong SELLING feeling 
     1 --> strong BUYING feeling
     """

    BUY_INDEX = np.zeros((len(data), 1))
    CLOSING_POSITION_CANDLE = np.zeros((len(data), 1))

    for index, row in enumerate(data):
        if index > len(data) - 60:
            continue

        max_price = np.amax(data[index:index + lookup][:, Constants.CANDLE_CLOSE])
        max_price_index = np.argmax(data[index:index + lookup][:, Constants.CANDLE_CLOSE])
        min_price = np.amin(data[index:index + lookup][:, Constants.CANDLE_CLOSE])
        min_price_index = np.argmin(data[index:index + lookup][:, Constants.CANDLE_CLOSE])

        current_buy_index = (row[Constants.CANDLE_CLOSE] - min_price) / (max_price - min_price)

        current_closing_position_candle = min_price_index

        BUY_INDEX[index][0] = current_buy_index
        CLOSING_POSITION_CANDLE[index][0] = current_closing_position_candle

    data_with_buy_index = np.hstack((data, BUY_INDEX))
    data_with_closing_position = np.hstack((data_with_buy_index, CLOSING_POSITION_CANDLE))
    n_last_removed = data_with_closing_position[0: len(data) - lookup]
    indexed_data = np.hstack((n_last_removed, np.arange(len(n_last_removed)).reshape(-1, 1)))
    return indexed_data
