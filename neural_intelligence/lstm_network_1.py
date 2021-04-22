import tensorflow as tf


class LstmNetwork1:
    def __init__(self):
        self.tf_inputs = tf.keras.Input(shape=(60,), batch_size=64)
        print(self.tf_inputs)
