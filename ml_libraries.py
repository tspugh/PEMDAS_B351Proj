import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier as mlp
from sklearn import svm as svm
from sklearn.model_selection import learning_curve as lc
import matplotlib.pyplot as plt
import torch
from data_harvester import spectra_reader as sr

# Loading the data
molecules = sr.load_data_both(debug=False)
# Creating a np array to extract data from the array of objects
molecules_array = np.array(molecules)
# Extracting the data from the array of objects
for i in range(len(molecules)):
    molecules_array[i] = molecules[i].monster_array
print(molecules_array.shape)
print(molecules_array)
print(molecules_array[0])
print(molecules_array[-1])
print(molecules_array[0][0])
y = []
X = []
for i in range(len(molecules_array)):
    y.append(molecules_array[i][0])
    X.append(molecules_array[i][1:])
print(y)
print(X)
print(len(X))
print(len(y))
y = np.array(y)
X = np.vstack(X)
print(X.shape)
print(X)

# Splitting the data into training and testing sets
# 80% of the data is used for training
# 20% of the data is used for testing
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# Prepare for 5-fold cross validation
# Split the training data into 5 folds
kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True)
accuracies = []
for train_index, test_index in kf.split(X_train):
    X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
    y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

    # Training the model
    clf = mlp(hidden_layer_sizes=(100), max_iter=1000)
    clf.fit(X_fold_train, y_fold_train)

    # Testing the model
    accuracy = clf.score(X_fold_test, y_fold_test)
    accuracies.append(accuracy)
    print(accuracy)
print(np.mean(accuracies))
clf = mlp(hidden_layer_sizes=(100), max_iter=1000)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
