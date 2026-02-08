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


class SpectralConv2d(nn.Module):
  def __init__(self, 
               in_channels: int, 
               out_channels: int, 
               modes1: int, 
               modes2: int):
    super().__init__()
    self.modes1 = modes1
    self.modes2 = modes2
    self.out_channels = out_channels
    self.scale = (2.0 / (in_channels + out_channels))**0.5
    self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
    self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

  def forward(self, x: torch.tensor):
    x_ft = torch.fft.rfft2(x)                               #[B, C, H, W//2 + 1]
    out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-2], x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
    # out_ft[:, :, :self.modes1, :self.modes2] = (x_ft[:,:,:self.modes1, :self.modes2].permute(2, 3, 0, 1) @ self.weights1.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
    # out_ft[:, :, -self.modes1:, :self.modes2] = (x_ft[:,:,-self.modes1:, :self.modes2].permute(2, 3, 0, 1) @ self.weights1.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
    # Multiply top-left corner (positive frequencies)
    out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
    # Multiply bottom-left corner (negative frequencies)
    out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy",x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
    return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
  def __init__(self,
               modes1: int,
               modes2: int,
               width: int,
               num_layers: int,
               padding_frac: float=0.25,
               in_channels: int=1,
               out_channels: int=2):
    super().__init__()
    self.modes1 = modes1
    self.modes2 = modes2
    self.width = width
    self.num_layers = num_layers
    self.padding_frac = padding_frac
    self.linear_p = nn.Linear(in_channels, width)
    self.conv_list = nn.ModuleList([
        nn.Conv2d(width, width, 1) for _ in range(num_layers)
    ])
    self.spec_list = nn.ModuleList([
        SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)
    ])
    self.norm_list = nn.ModuleList([
        nn.InstanceNorm2d(width) for _ in range(num_layers)
    ])
    self.linear_q = nn.Linear(width, 128)
    self.output_layer = nn.Linear(128, out_channels)
    self.activation = nn.GELU()

  def forward(self, x: torch.tensor):
    x = self.linear_p(x)                                    #[B, N, N, C] -> [B, N, N, W]
    x = x.permute(0, 3, 1, 2)                               #[B, W, N, N]
    x1_padding = int(round(x.shape[-1] * self.padding_frac))
    x2_padding = int(round(x.shape[-2] * self.padding_frac))
    x = nn.functional.pad(x, [0, x1_padding, 0, x2_padding])
    for spec, conv, norm in zip(self.spec_list, self.conv_list, self.norm_list):
      x1 = spec(x)
      x2 = conv(x)
      x = x1 + x2
      x = norm(x)
      x = self.activation(x)
    x = x[..., :x.size(-2)-x2_padding, :x.size(-1)-x1_padding]
    x = x.permute(0, 2, 3, 1)
    x = self.linear_q(x)
    x = self.activation(x)
    x = self.output_layer(x)
    amp_logit = x[..., 0]
    phase_logit = x[..., 1]
    amplitude = torch.sigmoid(amp_logit)
    phase = torch.tanh(phase_logit) * torch.pi
    return torch.stack([amplitude, phase], dim=-1)
