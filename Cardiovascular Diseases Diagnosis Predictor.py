import numpy as np
import pandas as pd
import math
from torch import nn
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ucimlrepo import fetch_ucirepo, list_available_datasets
import torch

'''
KNN for imaging, math proposal and understanding (done)
Contact the Hospital Professionals for treatment model (based on the same algorithm)
Add more Layers and Neurons to the current model (done)
Write Mathematical Foundation for first model (done)
Remove variables do the accuracy testing and see which ones seem to have the most influence
'''

'''
Git Hub

# Consider that the algorithm could update its KNN parameter to reduce a function that
indicates its accuracy. 

''' 

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes and then numpy arrays) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

X = X.to_numpy()
y = y.to_numpy() 

y = y[~np.isnan(X).any(axis = 1)]
X = X[~np.isnan(X).any(axis = 1)]
X = X/X.max(axis = 0) # Consider dividing by the maximum number column wise  

'''
# Has values different from just 0 and 1 which is strange
'''

y[y != 0] = 1

''' Take out the Cleveland Thing and see what happens with y being different from 0 and 1'''

n_samples, n_features = X.shape

# Data Preparation and Data Conversion to Arrays

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 1234) # Data Split 80/20

# Standarization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Model of 5 Layers
class NeuralNetwork(nn.Module):

    def __init__(self, n_input_features):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(n_input_features, 64)
        self.layer2 = nn.Linear(64,32)
        self.layer3 = nn.Linear(32,16)
        self.output = nn.Linear(16,1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = torch.sigmoid(self.output(x))
        return x

model = NeuralNetwork(n_features)

learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

num_epochs = 1000

for epoch in range(num_epochs):
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 20 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
    print(y_predicted)
    print(y_predicted_cls)