from builtins import print
import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt

from neural_intelligence.batches_generator import get_clean_data, generate_batch
from neural_intelligence.learning_rate_reducer_cb import LearningRateReducerCb


class LstmNetworkMultivariateStateless:

    def __init__(self, pair, sequence_length=60, n_future=60, features=10, univariate_index=0):
        self.slicing_train = 0.7 + (0.99-0.7)*random.random()
        self.univariate_index = univariate_index
        self.n_future = n_future

        self.full_dataset = get_clean_data(pair)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = self.scaler.fit(self.full_dataset)
        self.sequence_length = sequence_length
        self.features = features

        self.model = Sequential()
        self.model.add(LSTM(64, activation='relu', input_shape=(sequence_length, features),
                            return_sequences=True))
        self.model.add(LSTM(32, activation='relu', return_sequences=False))
        self.model.add(Dense(self.n_future))
        # used loss can be :
        # mean_squared_error, mean_absolute_error
        # optimizer can be :
        # adam, rmsprop
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()

    def predict(self):
        self.model.reset_states()

    def train(self):
        x, y = generate_batch(self.scaler.transform(self.full_dataset), self.univariate_index,
                              self.n_future,
                              sequence_length=self.sequence_length)
        print(x.shape, y.shape)
        self.model.fit(x, y, callbacks=[LearningRateReducerCb()], epochs=2,
                       validation_split=0.1, batch_size=16, verbose=1)

    def generate_random_slicing(self):
        self.slicing_train = 0.7 + (0.99 - 0.7) * random.random()

    def save(self):
        self.model.save('./neural_intelligence/models/lstm_network_multivariate_stateless'
                        '/lstm_network_multivariate_stateless.h5')

    def plot_forecast_vs_truth(self, pair, future=60):
        self.model = models.load_model('./neural_intelligence/models'
                                       '/lstm_network_multivariate_stateless'
                                       '/lstm_network_multivariate_stateless.h5')
        x = np.array(get_clean_data(pair))
        for i in range(20):
            self.generate_random_slicing()
            real_value = x[:int(len(x) * self.slicing_train + future)]
            real_value = real_value[-self.sequence_length - future:]
            real_value = real_value[:, self.univariate_index]
            data_to_use_for_prediction = x[:int(len(x) * self.slicing_train)]
            value_to_forecast_with = data_to_use_for_prediction[-self.sequence_length:]
            forecasted_value_to_plot = value_to_forecast_with[:, self.univariate_index]
            data_for_predictions = self.scaler.transform(value_to_forecast_with)
            data_for_predictions = np.expand_dims(data_for_predictions, axis=0)
            # Predictions
            predicted_value = self.model.predict(data_for_predictions)
            # for recurrence
            # BE VERY CAREFUL WITH INVERSE TRANSFOM, SELECT SAME INDEX, I LOST 10H ON THIS SHIT
            forecasted_value_to_plot = np.append(forecasted_value_to_plot,
                                                 self.scaler.inverse_transform(
                                                     np.repeat(
                                                         predicted_value.reshape(self.n_future, 1),
                                                         self.features, axis=-1))[:,
                                                 self.univariate_index].reshape(1, self.n_future))

            plt.plot(forecasted_value_to_plot, 'r--', label='predicted')
            plt.plot(real_value, label='real value')
            plt.legend(loc='best')
            plt.savefig("./neural_intelligence/LstmNetworkMultivariateStatelessData"
                        "/LstmNetworkMultivariateStateless"+str(i)+".png")
            plt.clf()
            plt.cla()
            plt.close()
