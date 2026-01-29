import os
import numpy as np
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from skimage.transform import resize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class PtychoDataset(Dataset):
  """
  Defines a PyTorch dataset for FNO models. The input to the model is has 3 channels and the 
  output has 2 chanels.
  Attributes:
    inputs: Inputs to the model of shape [N, S, S, 3], where the channels are the 
            diffraction pattern, local x grid point, local y grid point.
    targets: Model targets of shape [N, S, S, 2] where the channels are the 
            amplitude and the phase over the grid.
  """
  def __init__(self, 
               x_path, 
               intensity_target_path, 
               phase_target_path):
    """
    Initializes an instance of the Ptycho Dataset for the FNO Model.
    Args:
      x_path: path to npy file with the PtychoNN inputs of shape [N, 1, S, S].
      intensity_target_path: path to npy file with the intensity targets of shape [N, 1, S, S].
      phase_target_path: path to npy file with the phase targets of shape [N, 1, S, S].
    """
    x = torch.from_numpy(np.load(x_path)).to(torch.float).squeeze_(1)                       #[N, S, S]
    y_int = torch.from_numpy(np.load(intensity_target_path)).to(torch.float).squeeze_(1)    #[N, S, S]
    y_phi = torch.from_numpy(np.load(phase_target_path)).to(torch.float).squeeze_(1)        #[N, S, S]
    N, S = x.shape[0], x.shape[1]
    x_grid, y_grid = torch.meshgrid(torch.linspace(-1, 1, S), torch.linspace(-1, 1, S), indexing='xy')
    x_grid, y_grid = x_grid.unsqueeze_(0), y_grid.unsqueeze_(0)
    self.inputs = torch.stack([x, x_grid.expand(N, S, S), y_grid.expand(N, S, S)], dim=-1)  #[N, S, S, 3]
    self.targets = torch.stack([y_int, y_phi], dim=-1)                                      #[N, S, S, 2]

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx: int):
    return self.inputs[idx], self.targets[idx]

