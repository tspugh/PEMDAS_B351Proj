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
print("Shuffling the data")
np.random.shuffle(molecules_array)
y = []
X = []
for i in range(len(molecules_array)):
    y.append(molecules_array[i][0])
    X.append(molecules_array[i][1:])
print("Data loaded")
print("Labels:")
print(y)
print("Data:")
print(X)
print("Length of data:")
print(len(X))
print("Length of labels:")
print(len(y))
y = np.array(y)
X = np.vstack(X)
print("Shape of data:")
print(X.shape)

# # Normalizing the data
# X = sklearn.preprocessing.normalize(X)

# Splitting the data into training and testing sets
# 80% of the data is used for training
# 20% of the data is used for testing
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

print("Model 1")
clf = mlp(hidden_layer_sizes=(100), max_iter=1000)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy on test set:", accuracy)
print("Recall on test set:", sklearn.metrics.recall_score(y_test, clf.predict(X_test), average="macro"))
print("Precision on test set:", sklearn.metrics.precision_score(y_test, clf.predict(X_test), average="macro"))
print("F1 score on test set:", sklearn.metrics.f1_score(y_test, clf.predict(X_test), average="macro"))
plt.plot(clf.loss_curve_)
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.title("Loss curve for 500 hidden layer neurons")
plt.show()

print("Model 2")
clf = mlp(hidden_layer_sizes=(500), max_iter=1000)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy on test set:", accuracy)
print("Recall on test set:", sklearn.metrics.recall_score(y_test, clf.predict(X_test), average="macro"))
print("Precision on test set:", sklearn.metrics.precision_score(y_test, clf.predict(X_test), average="macro"))
print("F1 score on test set:", sklearn.metrics.f1_score(y_test, clf.predict(X_test), average="macro"))
plt.plot(clf.loss_curve_)
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.title("Loss curve for 500 hidden layer neurons")
plt.show()




