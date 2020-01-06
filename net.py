import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.lr = 0.01

    def train(self, x,y):
        self.layer1 = sigmoid(np.dot(x, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        self.backprop(x, y)
        return np.square(self.output - y).mean()
    def backprop(self, x, y):
        self.y = y
        self.input = x
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))
        self.weights1 += self.lr * d_weights1
        self.weights2 += self.lr * d_weights2

    def predict(self, x):
        self.layer1 = sigmoid(np.dot(x, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output
