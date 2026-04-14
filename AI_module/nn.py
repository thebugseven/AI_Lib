import numpy as np
from .math_utils import *


#==============================
# Activation Functions
#==============================

class activation_function:
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad_output):
        raise NotImplementedError("Backward method not implemented.")


class ReLU(activation_function):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)


class Sigmoid(activation_function):
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)


class Tanh(activation_function):
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)


class Softmax(activation_function):
    def __init__(self):
        self.output = None

    def forward(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        self.output = exp_x / np.sum(exp_x)
        return self.output

    def backward(self, grad_output):
        s = self.output.reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        return jacobian @ grad_output


#==============================
# Layers
#==============================

class Layer:
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad_output):
        raise NotImplementedError("Backward method not implemented.")
    
    def update(self, lr):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size, weights_init="xavier"):
        if weights_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (output_size, input_size))
        elif weights_init == "he":
            limit = np.sqrt(2 / input_size)
            self.W = np.random.randn(output_size, input_size) * limit
        else:
            self.W = np.random.randn(output_size, input_size) * 0.01

        self.b = np.zeros(output_size)

        self.input = None
        self.dW = None
        self.db = None

    def forward(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        self.input = x
        return self.W @ x + self.b
    
    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=float).reshape(-1)

        self.dW = np.outer(grad_output, self.input)
        self.db = grad_output.copy()
        grad_input = self.W.T @ grad_output
        return grad_input

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Activation(Layer):
    def __init__(self, activation):
        self.activation = activation

    def forward(self, x):
        return self.activation.forward(x)

    def backward(self, grad_output):
        return self.activation.backward(grad_output)


#==============================
# Losses
#==============================

class CrossEntropyLoss:
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(np.asarray(y_pred, dtype=float), self.eps, 1 - self.eps)
        self.y_true = np.asarray(y_true, dtype=float)
        return -np.sum(self.y_true * np.log(self.y_pred))

    def backward(self):
        return self.y_pred - self.y_true


#==============================
# Neural Network
#==============================

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def predict_proba(self, x):
        return self.forward(x)

    def predict(self, x):
        prob = self.forward(x)
        return int(np.argmax(prob))
