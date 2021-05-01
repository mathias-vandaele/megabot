import csv

import numpy as np
from sklearn.preprocessing import StandardScaler
from constants.constants import Constants


def generate_batch_yielded(pair, sequence_length=60):
    with open("resources/" + pair + ".csv", mode='r') as csv_file:
        data = np.array(list(csv.reader(csv_file)))
        clean_data = data[:, 1:11]
        for i in range(len(clean_data)-sequence_length-1):
            print(clean_data[i:i+sequence_length])
            print(clean_data[i+sequence_length+1][Constants.CANDLE_CLOSE])
            x_train = clean_data[i:i+sequence_length]
            y_train = clean_data[i+sequence_length+1, Constants.CANDLE_CLOSE]
        yield np.array(x_train), np.array(y_train)

def generate_batch(pair, sequence_length=60):
    with open("resources/" + pair + ".csv", mode='r') as csv_file:
        data = np.array(list(csv.reader(csv_file)))
        clean_data = data[:, [Constants.CANDLE_OPEN, Constants.CANDLE_HIGH, Constants.CANDLE_LOW, Constants.CANDLE_CLOSE, Constants.QUOTE_VOLUME]].astype(float)
        scaler = StandardScaler()
        scaler = scaler.fit(clean_data)
        clean_data_scaled = scaler.transform(clean_data)
        x_data, y_data = [], []
        for i in range(len(clean_data_scaled) - sequence_length - 1):
            x_data.append(clean_data_scaled[i:i + sequence_length])
            y_data.append(clean_data_scaled[i + sequence_length + 1, 3])
        return np.array(x_data), np.array(y_data)
