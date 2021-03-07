import torch
import numpy as np

def configure_loss_function(): 
  return torch.nn.BCEWithLogitsLoss()

def configure_optimizer(model):
  return torch.optim.SGD(model.parameters(), lr=.001)

def full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test, n_epochs=5000):
  train_losses = np.zeros(n_epochs)
  test_losses = np.zeros(n_epochs)

  for it in range(n_epochs): 
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    outputs_test = model(X_test)
    loss_test = criterion(outputs_test, y_test)

    train_losses[it] = loss.item()
    test_losses[it] = loss_test.item()

    if (it + 1) % 50 == 0:
      print(f'In this epoch {it+1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')

  return train_losses, test_losses
