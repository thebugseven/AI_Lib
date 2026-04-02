import numpy as np
from math_utils import *


class MLP:
	def __init__(self, input_size, hidden_size, output_size):
		self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(1 / input_size)
		self.b1 = np.zeros(hidden_size)

		self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(1 / hidden_size)
		self.b2 = np.zeros(output_size)

	def forward(self, x):
		x = np.array(x, dtype=float)

		z1 = self.W1 @ x + self.b1
		a1 = np.tanh(z1)

		z2 = self.W2 @ a1 + self.b2
		y_hat = softmax(z2)

		cache = {
			"x": x,
			"z1": z1,
			"a1": a1,
			"z2": z2,
			"y_hat": y_hat
		}
		return y_hat, cache
	
	def backward(self, cache, y_true):
		x = cache["x"]
		a1 = cache["a1"]
		y_hat = cache["y_hat"]
		y_true = np.array(y_true)

		delta2 = y_hat - y_true

		dW2 = np.outer(delta2, a1)
		db2 = delta2.copy()

		delta1 = (self.W2.T @ delta2) * (1 - a1 ** 2)

		dW1 = np.outer(delta1, x)
		db1 = delta1.copy()

		return dW1, db1, dW2, db2
	
	def update_params(self, dW1, db1, dW2, db2, lr):
		self.W1 -= lr * dW1
		self.b1 -= lr * db1
		self.W2 -= lr * dW2
		self.b2 -= lr * db2
