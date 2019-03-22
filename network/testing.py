import numpy as np
from activations import Activations

inputs = [[0, [0, 0]],
          [0, [0, 0]],
          [0, [0, 0]],
          [1, [0, 1]],
          [1, [0, 1]],
          [1, [0, 1]],
          [1, [1, 0]],
          [1, [1, 0]],
          [1, [1, 0]],
          [1, [1, 1]],
          [1, [1, 1]],
          [1, [1, 1]]]

input_features = [x[1] for x in inputs]
input_labels = [x[0] for x in inputs]

np.random.seed(1)

weights = 2 * np.random.random((2, 1)) - 1

print('Starting weights: ')
print(weights)

for _ in range(1):
    input_layer = input_features

    outputs = Activations.relu(np.dot(input_layer, weights))

print('Outputs after training')
print(outputs)
