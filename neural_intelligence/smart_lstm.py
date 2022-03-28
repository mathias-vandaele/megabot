import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential, models, Input, Model
from tensorflow.python.keras.layers import LSTM, Dense, Flatten
from tensorflow.python.keras.saving.saved_model.load import metrics

from neural_intelligence.batches_generator import generate_smart_lstm_batch, get_smart_lstm_data
from neural_intelligence.learning_rate_reducer_cb import LearningRateReducerCb


class SmartLSTM:

    def __init__(self, pair, sequence_length):
        self.sequence_length = sequence_length

        self.random_validation_slicing = 0.7 + (0.99 - 0.7) * random.random()
        self.full_dataset = get_smart_lstm_data(pair, lookback=30)
        self.training_data = self.full_dataset[:int(0.7 * len(self.full_dataset))]

        self.scalerI = MinMaxScaler(feature_range=(0, 1))
        self.scalerO = MinMaxScaler(feature_range=(0, 1))

        self.fitterI = self.scalerI.fit(self.training_data[:, [0, 3]])
        self.fitterO = self.scalerO.fit(self.training_data[:, [1, 2]])

        self.training_input = self.fitterI.transform(self.training_data[:, [0, 3]])
        self.training_output = self.fitterO.transform(self.training_data[:, [1, 2]])

        input_layer = Input(shape=(sequence_length, 2))
        layer_1_lstm = LSTM(128, return_sequences=True)(input_layer)

        layer_2_lstm_odd = LSTM(64, return_sequences=True)(layer_1_lstm)
        layer_3_lstm_odd = LSTM(32, return_sequences=True)(layer_2_lstm_odd)
        # guess the sell_index
        output_sell_index_proba = LSTM(1, activation='sigmoid')(layer_3_lstm_odd)

        layer_2_lstm_closing = LSTM(64, return_sequences=True)(layer_1_lstm)
        layer_3_lstm_closing = LSTM(32, return_sequences=True)(layer_2_lstm_closing)
        # guess when to close the position
        output_closing_position_candle = LSTM(1, activation='relu')(layer_3_lstm_closing)

        self.model = Model(inputs=input_layer, outputs=[output_sell_index_proba, output_closing_position_candle])

        self.model.compile(optimizer='adam',
                           loss={'lstm_3': 'mse', 'lstm_6': 'mse'}, metrics=[
                metrics.AUC()
            ])
        self.model.summary()

    def train(self):
        x, y1, y2 = generate_smart_lstm_batch(self.training_input, self.training_output, sequence_length=30)

        print(x.shape, y1.shape, y2.shape)

        self.model.fit(x, [y1, y2], callbacks=[LearningRateReducerCb()], epochs=10,
                       validation_split=0.1, batch_size=128, verbose=1)

    def save(self):
        self.model.save('./neural_intelligence/models/smart_lstm'
                        '/smart_lstm.h5')

    def generate_random_slicing(self):
        self.random_validation_slicing = 0.7 + (0.99 - 0.7) * random.random()

    def forecast_and_compare(self):
        self.model = models.load_model('./neural_intelligence/models/smart_lstm'
                                       '/smart_lstm.h5')
        for i in range(20):
            self.generate_random_slicing()
            validation_data = self.full_dataset[int(self.random_validation_slicing * len(self.full_dataset)): int(
                self.random_validation_slicing * len(self.full_dataset)) + self.sequence_length]
            to_predict_normalized = self.fitterI.transform(validation_data[:, [0, 3]])
            to_predict_reshaped = to_predict_normalized.reshape(1, 30, 2)
            prediction = self.model.predict(to_predict_reshaped)
            final_results = self.fitterO.inverse_transform(np.array(prediction).reshape(1, 2))

            print("FOUND BUY INDEX ==> " + str(final_results[0][0]))
            print("REAL BUY INDEX WAS ==> " + str(self.full_dataset[int((self.random_validation_slicing * len(
                self.full_dataset)) + self.sequence_length + 1)][1]))
            print("CLOSE POSITION IN ==> " + str(final_results[0][1]))

            if final_results[0][0] > 0.8:

                print("FOUND BUY INDEX ==> " + str(final_results[0][0]))
                print("REAL BUY INDEX WAS ==> " + str(self.full_dataset[int((self.random_validation_slicing * len(
                    self.full_dataset)) + self.sequence_length + 1)][1]))
                print("PRICE OF OPENING SHORT POSITION IS ==> " + str(self.full_dataset[int((
                                                                                                    self.random_validation_slicing * len(
                                                                                                self.full_dataset)) + self.sequence_length + 1)][
                                                                          0]))
                print("PRICE OF CLOSING SHORT POSITION IS ==> " + str(self.full_dataset[int((
                                                                                                    self.random_validation_slicing * len(
                                                                                                self.full_dataset)) + self.sequence_length + 1 + int(
                    final_results[0][1]))][0]))

                earned = self.full_dataset[
                    int((self.random_validation_slicing * len(self.full_dataset)) + self.sequence_length + 1)][0]
                earned -= self.full_dataset[
                    int((self.random_validation_slicing * len(self.full_dataset)) + self.sequence_length + 1 + int(
                        final_results[0][1]))][0]

                print("EARNED MONEY IS : " + str(earned) + "€")
                if earned > 0:
                    print("TRADE WON")
                else:
                    print("TRADE LOST")
