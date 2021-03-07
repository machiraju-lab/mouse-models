# -*- coding: utf-8 -*-
"""
Neural Linear Classifier

Date: 03-07-2021

Original file is located at
    https://colab.research.google.com/drive/1Pt_mGHcG_wep-mUfpV5PT4oE4EBopOHF
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from model import BinaryClassification
from train import *
from sklearn.model_selection import train_test_split

# Import data:
X = np.load("Linear-Classifier/data/neuron_data.npy", allow_pickle=True)
Y = np.load("Linear-Classifier/data/mouse_decision.npy", allow_pickle=True)

# Preprocessing:
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# This is optional to scale the neural data so that one neuron feature does not dominate:
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

_, input_dimension = X_train.shape

model = torch.nn.Linear(input_dimension, 1)



# Training:
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)

criterion = configure_loss_function()
optimizer = configure_optimizer(model)
train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test)

# To visualize the test/train loss:
# plt.plot(train_losses, label = 'train loss')
# plt.plot(test_losses, label = 'test loss')
# plt.legend()
# plt.show()



# Evaluation:
with torch.no_grad():
  p_train = model(X_train)
  p_train = (p_train.numpy() > 0)

  train_acc = np.mean(y_train.numpy() == p_train)

  p_test = model(X_test)
  p_test = (p_test.numpy() > 0)
  
  test_acc = np.mean(y_test.numpy() == p_test)

print("\n")
print(train_acc)
print(test_acc)