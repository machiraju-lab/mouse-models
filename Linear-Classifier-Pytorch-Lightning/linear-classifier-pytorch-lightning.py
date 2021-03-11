# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1Pt_mGHcG_wep-mUfpV5PT4oE4EBopOHF
"""

import torch
import pytorch_lightning as pl
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.core.lightning import LightningModule

X = np.load("Linear-Classifier/data/neuron_data.npy", allow_pickle=True)
Y = np.load("Linear-Classifier/data/mouse_decision.npy", allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data_tensor_train = torch.Tensor(X_train)
label_tensor_train = torch.Tensor(y_train)
dataset_train = data.TensorDataset(data_tensor_train, label_tensor_train)

data_tensor_test = torch.Tensor(X_test)
label_tensor_test = torch.Tensor(y_test)
dataset_test = data.TensorDataset(data_tensor_test, label_tensor_test)

loader_train = DataLoader(dataset_train)

loader_test = DataLoader(dataset_test)

# inputs, classes = next(iter(loader_train))
# print(inputs)

class MouseClassifier(pl.LightningModule):
  def __init__(self, input_dimension):
    super().__init__()
    self.linear = torch.nn.Linear(input_dimension, 1)

  def forward(self, input_dimension):
    return self.linear(input_dimension)

  def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=.001)

  def training_step(self, train_batch, batch_idx):
    x, y = train_batch
    outputs = self.linear(x)
    loss = torch.nn.BCEWithLogitsLoss()
    loss_val = loss(outputs[0], y)
    self.log('train_loss', loss_val, on_epoch=True)
    return loss_val
  
  def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    outputs = self.linear(x)
    loss = torch.nn.BCEWithLogitsLoss()
    loss_val = loss(outputs[0], y)
    self.log('val_loss', loss_val, on_epoch=True)

  def backward(self, trainer, loss_val, optimizer, optimizer_idx):
    loss.backward()

  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure, on_tpu, using_native_amp, using_lbfgs):
    optimizer.step()


# Test with Linear Regression instead: 

# model = LinearRegression(input_dim=554)
# trainer = pl.Trainer()
# trainer.fit(model, train_dataloader=loader_train, val_dataloaders=loader_test)
# trainer.test(test_dataloaders=loader_test)

model = MouseClassifier(554)



trainer = pl.Trainer(max_epochs=1)

trainer.fit(model, loader_train, loader_test)

with torch.no_grad():
  p_train = model(torch.Tensor(X_train))
  p_train = (p_train.numpy() > 0)

  train_acc = np.mean(torch.Tensor(y_train) == p_train)

  p_test = model(torch.Tensor(X_test))
  p_test = (p_test.numpy() > 0)
  
  test_acc = np.mean(torch.Tensor(y_test) == p_test)

print(train_acc)
print(test_acc)
