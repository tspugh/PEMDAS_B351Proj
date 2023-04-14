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

# Prepare for 5-fold cross validation
# Split the training data into 5 folds
kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True)
print("Model 1")
accuracies = []
count = 0
for train_index, test_index in kf.split(X_train):
    print("Fold", count + 1)
    X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
    y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

    # Training the model
    clf = mlp(hidden_layer_sizes=(100), max_iter=1000)
    clf.fit(X_fold_train, y_fold_train)

    # Testing the model
    accuracy = clf.score(X_fold_test, y_fold_test)
    accuracies.append(accuracy)
    print("Accuracy for fold", count + 1, ":", accuracy)
    count += 1
print("Average accuracy in 5-fold cross validation:", np.mean(accuracies))
clf = mlp(hidden_layer_sizes=(100), max_iter=1000)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy on test set:", accuracy)

print("Model 2")
accuracies = []
count = 0
for train_index, test_index in kf.split(X_train):
    print("Fold", count + 1)
    X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
    y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

    # Training the model
    clf = mlp(hidden_layer_sizes=(500), max_iter=1000)
    clf.fit(X_fold_train, y_fold_train)

    # Testing the model
    accuracy = clf.score(X_fold_test, y_fold_test)
    accuracies.append(accuracy)
    print("Accuracy for fold", count + 1, ":", accuracy)
    count += 1
print("Average accuracy in 5-fold cross validation:", np.mean(accuracies))
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

print("Model using PyTorch")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Convert the data to torch tensors
X_train = X[:int(len(X) * 0.8)]
X_test = X[int(len(X) * 0.8):]
y_train = y[:int(len(y) * 0.8)]
y_test = y[int(len(y) * 0.8):]
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()
# Convert the data to torch datasets
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
# Convert the data to torch dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
# Define the model
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(len(X[0]), 1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
net.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)



