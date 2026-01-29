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
    model: model to be trained
    train_dl: Dataloader with training data
    loss_fn: differentiable loss function
    opt: Optimizer to train model.
    device: Device on which model and data will reside.
  Returns:
      amplitude and phase reconstruction loss over the epoch.
  """
  model.train()
  amp_loss, phi_loss = 0.0, 0.0
  for xb, yb in train_dl:
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()
    out = model(xb)
    loss_amp = loss_fn(out[..., 0], yb[..., 0])
    loss_phi = loss_fn(out[..., 1], yb[..., 1])
    loss = loss_amp + loss_phi
    loss.backward()
    opt.step()
    amp_loss += loss_amp.detach().item()
    phi_loss += loss_phi.detach().item()
  return amp_loss / len(train_dl), phi_loss / len(train_dl)


def val_step(model, 
             val_dl, 
             loss_fn, 
             device):
  """
  Performs evaluation model on validation dataloader,
  returning model loss on the amplitude and phase reconstruction.
  Args:
    model: model to be trained
    val_dl: Dataloader with evaluation data
    loss_fn: loss function
    device: Device on which model and data will reside.
  Returns:
      amplitude and phase reconstruction loss over the epoch.
  """
  model.eval()
  amp_loss, phi_loss = 0.0, 0.0
  with torch.inference_mode():
    for xb, yb in val_dl:
      xb, yb = xb.to(device), yb.to(device)
      out = model(xb)
      loss_amp = loss_fn(out[..., 0], yb[..., 0])
      loss_phi = loss_fn(out[..., 1], yb[..., 1])
      amp_loss += loss_amp.item()
      phi_loss += loss_phi.item()
  return amp_loss / len(val_dl), phi_loss / len(val_dl)


def train(model, 
          train_dl, 
          val_dl, 
          loss_fn, 
          opt, 
          device, 
          num_epochs=50,
          scheduler=None):
  """
  Performs defined number of epochs of training and evaluation for the model on
  the data loaders, returning the loss histories.
  Args:
    model: model to be trained and evaluated.
    train_dl: Dataloader with training data.
    val_dl: Dataloader with testing data.
    opt: Optimizer to tune model params.
    device: Device on which model and eventually data shall reside.
    num_epochs: Number of epochs of training.
    scheduler: Learning rate scheduler.
  Returns:
    tuple of train and validation loss histories, over amplitude and phase.
  """
  train_amp_losses, train_phi_losses, val_amp_losses, val_phi_losses = [], [], [], []
  for epoch in range(num_epochs):
    train_amp_loss, train_phi_loss = train_step(model, train_dl, loss_fn, opt, device)
    val_amp_loss, val_phi_loss = val_step(model, val_dl, loss_fn, device)
    train_amp_losses.append(train_amp_loss)
    train_phi_losses.append(train_phi_loss)
    val_amp_losses.append(val_amp_loss)
    val_phi_losses.append(val_phi_loss)
    if scheduler:
      scheduler.step(val_phi_loss)
    print(f"Epoch: {epoch+1} Train Amp Loss: {train_amp_losses[-1]:.5f} Train Phi Loss: {train_phi_losses[-1]:.5f} Val Amp Loss: {val_amp_losses[-1]:.5f} Val Phi Loss: {val_phi_losses[-1]:.5f}")
  return train_amp_losses, train_phi_losses, val_amp_losses, val_phi_losses


