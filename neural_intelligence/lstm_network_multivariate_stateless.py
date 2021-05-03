from builtins import print

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout, LSTM, Dense
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt

from neural_intelligence.batches_generator import get_clean_data, generate_batch
from neural_intelligence.learning_rate_reducer_cb import LearningRateReducerCb


class LstmNetworkMultivariateStateless:

    def __init__(self, pair,  sequence_length=60, features=10):
        self.slicing_train = 0.9
        self.univariate_index = 0

        self.full_dataset = get_clean_data(pair)
        self.scaler = StandardScaler()
        self.scaler = self.scaler.fit(self.full_dataset)
        self.sequence_length = sequence_length
        self.features = features

        self.model = Sequential()
        self.model.add(LSTM(64, activation='relu', input_shape=(sequence_length, features),
                            return_sequences=True))
        self.model.add(LSTM(32, activation='relu', return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
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
                              sequence_length=self.sequence_length)
        self.model.fit(x, y, callbacks=[LearningRateReducerCb()], epochs=10,
                       validation_split=0.1, batch_size=16, verbose=1)

    def save(self):
        self.model.save('./neural_intelligence/models/lstm_network_multivariate_stateless'
                        '/lstm_network_multivariate_stateless.h5')

    def plot_forecast_vs_truth(self, pair, future=60):
        self.model = models.load_model('./neural_intelligence/models'
                                       '/lstm_network_multivariate_stateless'
                                       '/lstm_network_multivariate_stateless.h5')
        x = np.array(get_clean_data(pair))
        real_value = x[:int(len(x) * self.slicing_train + future)]
        real_value = real_value[-self.sequence_length - future:]
        real_value = real_value[:, self.univariate_index]
        y = x[:int(len(x) * self.slicing_train)]
        value_to_forecast_with = y[-self.sequence_length:]
        forecasted_value_to_plot = value_to_forecast_with[:, self.univariate_index]
        print(forecasted_value_to_plot.shape, real_value.shape)

        data_for_predictions = self.scaler.transform(value_to_forecast_with)
        data_for_predictions = np.expand_dims(data_for_predictions, axis=0)
        # Predictions
        predicted_value = self.model.predict(data_for_predictions)
        # for recurrence
        forecasted_value_to_plot = np.append(forecasted_value_to_plot, self.scaler.inverse_transform(np.repeat(predicted_value, self.features, axis=-1))[:, 0])

        plt.plot(forecasted_value_to_plot)
        plt.plot(real_value)
        plt.show()
