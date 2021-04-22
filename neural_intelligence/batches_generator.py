import csv

import numpy as np


def generate_batch(pair, batch_size=64, sequence_length=3):
    with open("resources/" + pair + ".csv", mode='r') as csv_file:
        inputs = list(csv.reader(csv_file))
        attributes = len(inputs[0])
        # size of chunk
        chuck_size = (len(inputs) - 1) // batch_size
        # by dividing the chuck by the sequence length, we have the total number of batches
        number_of_batches = chuck_size // sequence_length
        print(chuck_size, number_of_batches)
        for s in range(0, number_of_batches):
            batch_inputs = np.zeros((batch_size, sequence_length, attributes))
            batch_targets = np.zeros((batch_size, sequence_length, attributes))
            for b in range(0, batch_size):
                fr = (b * chuck_size) + (s * sequence_length)
                to = fr + sequence_length
                batch_inputs[b] = inputs[fr:to]
                batch_targets[b] = inputs[fr + 1:to + 1]
            yield batch_inputs, batch_targets
