from tensorflow.keras.layers import Dropout, LSTM, Dense
from tensorflow.keras.models import Sequential


class LstmNetworkMultivariateStateful:
    def __init__(self, sequence_length=60, features=10):
        # TODO : implement a stateful model ==> is a 3D input instead of 2D. Why ? Because we
        #  will repeat looking forward the next price without resetting the state of the cell.
        #  Here is an example; I want to look forward 60 minutes in advance, and looking at the
        #  last 2h (120m) backward then for the first prediction, I'll use an entry sequence of
        #  120, and the size of the batch will be 60, that will give us a matrix of (60, 120,
        #  nb_features). As a training entry It could be the 4D matrix (X, 60, 120, nb_features)

        self.model = Sequential()
        # TODO : To be implemented
        self.model.summary()

    def predict(self):
        self.model.reset_states()

    def train(self, x, y):
        history = self.model.fit(x, y, epochs=4, batch_size=16, validation_split=0.2, verbose=1)
        print(history)

