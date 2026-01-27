import torch
import torch.nn as nn


class SpectralConv1D(nn.Module):
  def __init__(self, 
               in_channels: int, 
               out_channels: int, 
               modes: int):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.modes = modes
    self.scale = 1.0 / (in_channels * out_channels)
    self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

  def forward(self, x):
    x_ft = torch.fft.rfft(x) #[B, C, N: Num Grid Points] -> [B, C, N//2 + 1]
    out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
    out_ft[:, :, :self.modes] = (x_ft[:,:,:self.modes].permute(2, 0, 1) @ self.weights.permute(2, 0, 1)).permute(1, 2, 0)
    return torch.fft.irfft(out_ft, n=x.shape[-1])


class FNO1d(nn.Module):
  def __init__(self, 
               modes: int, 
               width: int):
    super().__init__()
    self.linear_p = nn.Linear(2, width)
    self.spec1 = SpectralConv1d(width, width, modes)
    self.lin1 = nn.Conv1d(width, width, 1)
    self.spec2 = SpectralConv1d(width, width, modes)
    self.lin2 = nn.Conv1d(width, width, 1)
    self.spec3 = SpectralConv1d(width, width, modes)
    self.lin3 = nn.Conv1d(width, width, 1)
    self.linear_q = nn.Linear(width, 32)
    self.output_layer = nn.Linear(32, 1)
    self.activation = nn.Tanh()

  def forward(self, x: torch.tensor):
    #x = [batch size, num grid points (N), 2]
    x = self.linear_p(x)                              #[batchsize, N, width]
    x = x.permute(0, 2, 1)                            #[batchsize, width, N]
    x = self.activation(self.spec1(x) + self.lin1(x)) #[batchsize, width, N]
    x = self.activation(self.spec2(x) + self.lin2(x)) #[batchsize, width, N]
    x = self.activation(self.spec3(x) + self.lin3(x)) #[batchsize, width, N]
    x = x.permute(0, 2, 1)                            #[batchsize, N, width]
    x = self.activation(self.linear_q(x))             #[batchsize, N, 32]
    return self.output_layer(x)                       #[batchsize, N, 1]


