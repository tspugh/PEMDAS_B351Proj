import numpy as np
import pandas as pd
from data_harvester import spectra_reader as sr

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs):
        # input_size = number of features
        self.input_size = input_size
        # hidden_size = number of neurons in the hidden layer
        self.hidden_size = hidden_size
        # output_size = number of classes
        self.output_size = output_size
        # learning_rate = learning rate
        self.learning_rate = learning_rate
        # hidden_weights = weights between the input layer and the hidden layer
        self.hidden_weights = np.random.rand(self.input_size, self.hidden_size)
        # hidden_bias = bias for the hidden layer
        self.hidden_bias = np.random.rand(self.hidden_size, 1)
        # output_weights = weights between the hidden layer and the output layer
        self.output_weights = np.random.rand(self.hidden_size, self.output_size)
        # output_bias = bias for the output layer
        self.output_bias = np.random.rand(self.output_size, 1)
        # hidden_layer = hidden layer values
        self.hidden_layer = np.zeros((1, self.hidden_size))
        # output_layer = output layer values
        self.output_layer = np.zeros((1, self.output_size))
        # sse = sum of squared errors
        self.sse = 0
        # epochs = number of epochs
        self.epochs = epochs

    # sigmoid activation function for the hidden layer
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # sigmoid derivative for the backpropagation algorithm
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # relu activation function for the hidden layer (not used, but can be used)
    def relu(self, x):
        return np.maximum(0, x)

    # relu derivative for the backpropagation algorithm (not used, but can be used)
    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)

    # softmax activation function for the output layer
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    # backpropagation algorithm
    def backpropagation_algorithm(self, X, y):
        # forward propagation
        # hidden layer
        for j in range(self.hidden_size):
            h = 0
            for i in range(self.input_size):
                h += X[i] * self.hidden_weights[i][j]
            h += self.hidden_bias[j]
            self.hidden_layer[0][j] = self.sigmoid(h)
        output_h_values = np.zeros((1, self.output_size))
        # output layer
        for k in range(self.output_size):
            h = 0
            for j in range(self.hidden_size):
                h += self.hidden_layer[0][j] * self.output_weights[j][k]
            h += self.output_bias[k]
            output_h_values[0][k] = h
        self.output_layer = self.softmax(output_h_values)
        # error calculation
        self.sse = 0.5 * (np.sum(self.output_layer - y) * np.sum(self.output_layer - y))
        # backpropagation
        # output layer
        new_output_weights = np.zeros((self.hidden_size, self.output_size))
        new_output_bias = np.zeros((self.output_size, 1))
        for k in range(self.output_size):
            delta_k = (self.output_layer[0][k] - y[k]) * self.sigmoid_derivative(self.output_layer[0][k])
            for j in range(self.hidden_size):
                new_output_weights[j][k] = self.output_weights[j][k] - self.learning_rate * delta_k * self.hidden_layer[0][j]
            new_output_bias[k][0] = self.output_bias[k][0] - self.learning_rate * delta_k
        # hidden layer
        new_hidden_weights = np.zeros((self.input_size, self.hidden_size))
        new_hidden_bias = np.zeros((self.hidden_size, 1))
        for j in range(self.hidden_size):
            delta_j = np.dot(self.hidden_layer[0][j] * (1 - self.hidden_layer[0][j]), np.sum((self.output_layer[0][k] - y[k]) * self.output_weights[j][k] * (1 - self.output_layer[0][k]) * self.output_weights[j][k] for k in range(self.output_size)))
            for i in range(self.input_size):
                new_hidden_weights[i][j] = self.hidden_weights[i][j] - self.learning_rate * delta_j * X[i]
            new_hidden_bias[j][0] = self.hidden_bias[j][0] - self.learning_rate * delta_j
        self.output_weights = new_output_weights
        self.output_bias = new_output_bias
        self.hidden_weights = new_hidden_weights
        self.hidden_bias = new_hidden_bias

    # training the neural network
    def train(self, X, y):
        # training for epochs
        for epoch in range(self.epochs):
            # training for each sample
            for i in range(len(X)):
                self.backpropagation_algorithm(X[i], y[i])
            # printing the accuracy every 50 epochs
            if epoch % 50 == 0:
                print("Epoch: ", epoch)
                print("Accuracy: ", self.accuracy(X, y))

    # predicting the output for a given input
    def predict(self, X):
        # predicting the output for a given input
        for j in range(self.hidden_size):
            h = 0
            for i in range(self.input_size):
                h += X[i] * self.hidden_weights[i][j]
            h += self.hidden_bias[j]
            self.hidden_layer[0][j] = self.sigmoid(h)
        output_h_values = np.zeros((1, self.output_size))
        for k in range(self.output_size):
            h = 0
            for j in range(self.hidden_size):
                h += self.hidden_layer[0][j] * self.output_weights[j][k]
            h += self.output_bias[k]
            output_h_values[0][k] = h
        self.output_layer = self.softmax(output_h_values)
        # returning the index of the maximum value in the output layer
        # this then corresponds to the predicted class
        for i in range(len(self.output_layer[0])):
            if self.output_layer[0][i] == max(self.output_layer[0]):
                return i + 1

    # calculating the accuracy of the model
    def accuracy(self, X, y):
        # counting the number of correct predictions
        correct = 0
        for i in range(len(X)):
            # if the predicted class is equal to the actual class
            if self.predict(X[i]) == np.argmax(y[i]) + 1:
                correct += 1
        return correct / len(X)


if __name__ == "__main__":
    # Loading the data
    molecules = sr.load_data_both(debug=False)
    # Creating a np array to extract data from the array of objects
    molecules_array = np.array(molecules)
    # Extracting the data from the array of objects
    for i in range(len(molecules)):
        molecules_array[i] = molecules[i].monster_array
    print(molecules_array.shape)
    print(molecules_array[0])
    print(molecules_array[-1])
    y = []
    X = []
    for i in range(len(molecules_array)):
        y.append([molecules_array[i][0]])
        X.append(molecules_array[i][1:])
    # Converting the data into numpy arrays
    y = np.array(y)
    X = np.vstack(X)
    print(X.shape)
    print(y.shape)
    print(X)
    print(y)
    # Splitting the data into training and testing sets
    X_train = X[:int(0.8 * len(X))]
    X_test = X[int(0.8 * len(X)):]
    y_train = y[:int(0.8 * len(y))]
    y_test = y[int(0.8 * len(y)):]
    mlp = MLP(len(X[0]), 100, 1, learning_rate=0.01, epochs=1000)
    mlp.train(X_train, y_train)
    print("Accuracy on the test set: " + str(mlp.accuracy(X_test, y_test)))



