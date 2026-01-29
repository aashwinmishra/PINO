import torch
import torch.nn as nn


def train_step(model, 
               train_dl, 
               loss_fn, 
               opt, 
               device):
  """
  Performs 1 epoch of training of model on train dataloader,
  returning model loss on the amplitude and phase reconstruction.
  Args:
    model: model too be trained
    train_dl: Dataloader with training data
    loss_fn: differentiable loss function
    opt: Optimizer to train model.
    device: Device on which model and data will reside.
  Returns:
      loss over the epoch.
  """
  model.train()
  epoch_loss = 0.0
  for xb, yb in train_dl:
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()
    out = model(xb)
    loss = loss_fn(out, yb)
    loss.backward()
    opt.step()
    epoch_loss += loss.cpu().item()
  return epoch_loss / len(train_dl)


def val_step(model, 
             val_dl, 
             loss_fn, 
             device):
  """
  Performs evaluation model on validation dataloader,
  returning model loss on the amplitude and phase reconstruction.
  Args:
    model: model too be trained
    val_dl: Dataloader with evaluation data
    loss_fn: loss function
    device: Device on which model and data will reside.
  Returns:
      loss over the epoch.
  """
  model.eval()
  epoch_loss = 0.0
  with torch.inference_mode():
    for xb, yb in val_dl:
      xb, yb = xb.to(device), yb.to(device)
      out = model(xb)
      loss = loss_fn(out, yb)
      epoch_loss += loss.cpu().item()
  return epoch_loss / len(val_dl)


def train(model, 
          train_dl, 
          val_dl, 
          loss_fn, 
          opt, 
          device, 
          num_epochs):
  """
  Performs defined number of epochs of training and evaluation for the model on
  the data loaders, returning the loss histories.
  Args:
    model: model to be trained and evaluated.
    train_dl: Dataloader with training data.
    val_dl: Dataloader with testing data.
    opt: Optimizer to tune model params.
    device: Device on which model and eventually data shall reside
    num_epochs: Number of epochs of training
  Returns:
    tuple of train and validation loss histories.
  """
  train_losses, val_losses = [], []
  for epoch in range(num_epochs):
    train_loss = train_step(model, train_dl, loss_fn, opt, device)
    val_loss = val_step(model, val_dl, loss_fn, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch: {epoch+1} Train Loss: {train_losses[-1]:.5f} Val Loss: {val_losses[-1]:.5f}")
  return train_losses, val_losses
      

