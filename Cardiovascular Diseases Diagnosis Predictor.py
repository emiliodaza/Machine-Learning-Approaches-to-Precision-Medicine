import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from torch import nn
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from ucimlrepo import fetch_ucirepo, list_available_datasets
import torch
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns

seed = 27
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df = pd.read_csv("processed.cleveland.data", header = None, names = column_names, na_values = "?") # header = None means that there is no header row (which would be a row containing column names) but instead all rows contain data, names = column_names means that that list instead would be the header containing the column names, na_values = ? insdicates that any ? in the file should be interpreted as a missing value
df = df.dropna() # removes rows from the dataframe that contain missing values ("?")

X = df.drop(columns = ["target"]).values # drops the column named "target" and turns the new dataframe into a NumPy array
y = df["target"].values # creates a variable y composed of a NumPy array that contains the values of the column "target" in the original dataframe

#X = X/X.max(axis = 0) Finds the maximum value for each column/feature of the NumPy array X and creates a 1D array that has the maximum value for each feature in order and uses this vector to divide each entry of X to the corresponding maximum for its feature

y[y != 0] = 1 # turns any element in the 1D array that is different from 0 into 1 (0 = absence or 1 = presence)

n_samples, n_features = X.shape # rows represent samples and columns features

# First Split: 80% training, 20% temporary (temp) which will be further split into validation and testing datasets

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = 1234) # random_state = 1234 guarantees that the mode is always trained with the same data. Applies for validation and testing dataset too!

# Second split: 10% validation, 10% test from the temporary split (which is 20% of the original data)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 1234)

# Standarization
sc = StandardScaler() # This class fromt he sklearn.preprocessing module allows to scale features to a common range using the standarization formula
X_train = sc.fit_transform(X_train) # Calculates the mean and standard deviation for the each of the 13 features individually, not for the whole X_train. And will accordingly use the formula and mean and standard deviation to apply the standarization formula for each entry with respect to its corresponding feature.
X_val = sc.transform(X_val) # X_val is the input of the validation dataset
X_test = sc.transform(X_test) # sc.transform(X_val) and sc.transform(X_test) will perform the same operations for the mentioned partitions of the dataset however the means and standard deviations per feature won't be recalculated rather re-used from X_train

X_train = torch.from_numpy(X_train.astype(np.float32)) # Transforms the NumPy array X_train to another of type float32 so that each number in the array will be represented by 32 bits of memory letting numbers between approximately -3.4e38 and 3.4e38 be used. It also transforms the NumPY array into a PyTorch tensor
X_val = torch.from_numpy(X_val.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
y_val = torch.from_numpy(y_val.astype(np.float32)).view(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)

print(f"Number of samples in the training dataset: {len(X_train)}")
print(f"Number of samples in the validation dataset: {len(X_val)}")
print(f"Number of samples in the testing dataset: {len(X_test)}")

total_samples = len(X_train) + len(X_val) + len(X_test)
train_percentage = (len(X_train) / total_samples) * 100
val_percentage = (len(X_val) / total_samples) * 100
test_percentage = (len(X_test) / total_samples) * 100

print(f"Training dataset: {train_percentage:.2f}% of total data")
print(f"Validation dataset: {val_percentage:.2f}% of total data")
print(f"Testing dataset: {test_percentage:.2f}% of total data")

# Model of 5 Layers
class NeuralNetwork(nn.Module): # the created class Neural Network inherits from the parent class nn.Module

    def __init__(self, n_input_features): # the constructor method is called that takes in an instance and also the number of input features
        super(NeuralNetwork, self).__init__() # this allows us to access the __init_ method inside of the parent class which is nn.Module
        self.layer1 = nn.Linear(n_input_features, 360) # the first argument is the amount of input_neurons and the second is the number of output neurons. The same applies for the rest
        self.layer2 = nn.Linear(360,180)
        self.layer3 = nn.Linear(180,90)
        self.layer4 = nn.Linear(90,45)
        self.output = nn.Linear(45,1)
        self.relu = nn.ReLU() # activation function for the hidden layers
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = torch.sigmoid(self.output(x)) # final activation function for the output layer
        return x

model = NeuralNetwork(n_features)

learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
num_epochs = 482

accuracies_train = []
accuracies_val = []
ci_margins_95_val = []
epochs = []

final_epoch_val_accuracy = None
final_epoch_test_accuracy = None

for epoch in range(num_epochs):
    model.train() #enables gradient tracking (have a version that expresses everything in terms of variables)
    y_predicted = model(X_train) # returns the output of the neural network using the training dataset input part
    loss = criterion(y_predicted, y_train) # applies the Binary Cross Entropy Loss using y_predicted and y-train (true labels)
    
    loss.backward() # Calculates the gradients of BCE w.r.t the model's parameters (weights and biases)
    optimizer.step() # updates the model parameters using Stochastic Gradient Descend because that was the one specified
    optimizer.zero_grad() # the gradient calculated becomes 0 after used because otherwise it would get added together with the next epoch's gradient

    model.eval()
    with torch.no_grad():
        # Training accuracy
        y_predicted_train = model(X_train)
        y_predicted_class_train = y_predicted_train.round()
        overall_acc_train = y_predicted_class_train.eq(y_train).sum().item() / float(y_train.shape[0])
        accuracies_train.append(overall_acc_train)
        # Validation accuracy
        y_predicted_val = model(X_val)
        y_predicted_class_val = y_predicted_val.round()
        overall_acc_val = y_predicted_class_val.eq(y_val).sum().item() / float(y_val.shape[0])

        ci_margin_95_val = 1.96 * (overall_acc_val * (1 - overall_acc_val) / y_val.shape[0]) ** 0.5

        accuracies_val.append(overall_acc_val)
        ci_margins_95_val.append(ci_margin_95_val)
        epochs.append(epoch + 1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}, "
                  f"Validation Accuracy: {overall_acc_val * 100:.2f}%, "
                  f"Training Accuracy: {overall_acc_train * 100:.2f}%, "
                  f"95% CI Margin: {ci_margin_95_val:.4f}")
        if epoch == num_epochs - 1:
            last_y_predicted_val = y_predicted_class_val
            last_confusion_matrix_val = confusion_matrix(y_val.numpy(), last_y_predicted_val.numpy())

            last_y_predicted_test = model(X_test).round()
            last_confusion_matrix_test = confusion_matrix(y_test.numpy(), last_y_predicted_test.numpy())            

# Graphing

print(f"\nFinal Validation Accuracy (Last Epoch): {accuracies_val[-1] * 100:.2f}%")
sns.heatmap(last_confusion_matrix_val, annot = True, fmt = "d", cmap = "Blues")
plt.title("Confusion Matrix - Validation Dataset")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(f"Final Testing Accuracy: {last_y_predicted_test.eq(y_test).sum().item() / float(y_test.shape[0]) * 100:.2f}%")
sns.heatmap(last_confusion_matrix_test, annot = True, fmt = "d", cmap = "Blues")
plt.title("Confusion Matrix - Testing Dataset")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

plt.figure(figsize = (10, 6))
plt.plot(epochs, [100 * acc for acc in accuracies_train], marker = "o", label = "Training Accuracy", color = "blue")
plt.plot(epochs, [100 * acc for acc in accuracies_val], marker = "o", label = "Validation Accuracy", color = "purple")
best_epoch_val = np.argmax(accuracies_val)
best_accuracy_val = accuracies_val[best_epoch_val]
best_ci_margin_95_val = ci_margins_95_val[best_epoch_val]

plt.errorbar(epochs[best_epoch_val], 100 * best_accuracy_val, yerr = 100 * best_ci_margin_95_val, fmt = "o", color = "green", ecolor = "black", capsize = 5, label = "Best Validation Epoch with 95% CI Margin")
plt.axhline(y = 100 * (best_accuracy_val + best_ci_margin_95_val), color = "black", linestyle = "--")
plt.axhline(y = 100 * (best_accuracy_val - best_ci_margin_95_val), color = "black", linestyle = "--")

plt.annotate(f"Best: Epoch {best_epoch_val + 1}, {100 * best_accuracy_val:.2f}%",
             xy = (epochs[best_epoch_val], 100 * best_accuracy_val),
             xytext = (epochs[best_epoch_val] + 50, 100 * best_accuracy_val - 5),
             arrowprops = dict(facecolor = "red", arrowstyle = "->"),
             fontsize = 10, color = "green")

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Training and Validation Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.show()