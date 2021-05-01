from tensorflow.keras.layers import Dropout, LSTM, Dense
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt

from neural_intelligence.learning_rate_reducer_cb import LearningRateReducerCb


class LstmNetworkMultivariateStateless:
    def __init__(self, sequence_length=60, features=10):
        self.model = Sequential()
        self.model.add(LSTM(128, activation='relu', input_shape=(sequence_length, features),
                            return_sequences=True))
        self.model.add(LSTM(32, activation='relu', return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()

    def predict(self):
        self.model.reset_states()

    def train(self, x, y):
        history = self.model.fit(x, y, callbacks=[LearningRateReducerCb()], epochs=30, batch_size=16, validation_split=0.1, verbose=1)
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.show()

