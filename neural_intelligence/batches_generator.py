import csv

import numpy as np
from constants.constants import Constants


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


def get_clean_data(pair):
    with open("resources/" + pair + ".csv", mode='r') as csv_file:
        data = np.array(list(csv.reader(csv_file)))
        clean_data = data[:, [Constants.CANDLE_CLOSE, Constants.QUOTE_VOLUME, Constants.TAKER_BUY_ASSET_VOLUME]].astype(float)
        return clean_data
