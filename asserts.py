import numpy as np
from project import errors, activations, network

tolerance = 1e-6

real = np.array([1, 2, 3.2, -1, -2.3])
forecasted = np.array([1.1, 1.99, 3.2, -0.9, -2.2])

try:
	correct = 0.077588659
	assert(np.abs(errors.mse(real, forecasted) - correct) <= tolerance)
	print("MSE PASSED!")
except:
	print("MSE NOT PASSED!")

positive = 5
negative = -5
correct_positive = 0.99330714907
correct_negative = 0.00669285092

try:
	assert(np.abs(activations.sigmoid(positive) - correct_positive) <= tolerance)
	print("SIGMOID POSITIVE PASSED!")

	assert(np.abs(activations.sigmoid(negative) - correct_negative) <= tolerance)
	print("SIGMOID NEGATIVE PASSED!")
except:
	print("SIGMOID NOT PASSED!")

positive = 5
negative = -5
correct_positive = -20
correct_negative = -30

try:
	assert(activations.sigmoid_derivative(positive) == correct_positive)
	print("SIGMOID DERIVATIVE POSITIVE PASSED!")

	assert(activations.sigmoid_derivative(negative) == correct_negative)
	print("SIGMOID DERIVATIVE NEGATIVE PASSED!")
except:
	print("SIGMOID DERIVATIVE NOT PASSED!")

inputs = np.array([
	[1.0, 0.0, 0.0],
	[1.0, 0.0, 1.0],
	[1.0, 1.0, 0.0],
	[1.0, 1.0, 1.0]
])

targets = np.array([
	[0.0],
	[1.0],
	[1.0],
	[0.0],
])

net = network(inputs, targets)

try:
	assert(np.array_equal(inputs, net.inputs))
	print("NET INPUTS PASSED!")

	assert(np.array_equal(targets, net.targets))
	print("NET TARGETS PASSED!")

	assert(net.hidden_layer_neurons == 2)
	print("HIDDEN LAYER NEURONS PASSED!")

	assert(net.learning_rate == 0.5)
	print("LEARNING RATE PASSED!")

	assert(net.weights_layer_1.shape == (3, 3))
	print("WEIGHTS 1 PASSED!")

	assert(net.weights_layer_2.shape == (3, 1))
	print("WEIGHTS 2 PASSED!")
except:
	print("NETWORK NOT PASSED!")

print("TRAINING NETWORK")

epochs = 10000
net.fit(epochs)

predicted = net.predict(inputs)
track = (np.abs(predicted - targets) <= 0.2).all(axis = 1)

try:
	assert(np.array_equal(track, np.array([True, True, True, True])))
	print("PREDICT PASSED!")
except:
	print("PREDICT NOT PASSED!")