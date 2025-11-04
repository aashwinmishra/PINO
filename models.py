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
