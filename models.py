import torch
import torch.nn as nn


class SpectralConv1D(nn.Module):
  """
  1D spectral convolution layer for use in FNO architectures.
  Applies a linear transformation in the truncated Fourier domain by
  multiplying the lowest `modes` Fourier coefficients with learned complex
  weights, then transforming back to the spatial domain via irfft.
  """

  def __init__(self,
               in_channels: int,
               out_channels: int,
               modes: int):
    """
    Initializes a SpectralConv1D layer.
    Args:
      in_channels: Number of input feature channels.
      out_channels: Number of output feature channels.
      modes: Number of lowest Fourier modes to retain and learn.
    """
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.modes = modes
    self.scale = 1.0 / (in_channels * out_channels)
    self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the 1D spectral convolution.
    Args:
      x: Input tensor of shape [B, C_in, N].
    Returns:
      Real-valued output tensor of shape [B, C_out, N].
    """
    x_ft = torch.fft.rfft(x) #[B, C, N: Num Grid Points] -> [B, C, N//2 + 1]
    out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
    out_ft[:, :, :self.modes] = (x_ft[:,:,:self.modes].permute(2, 0, 1) @ self.weights.permute(2, 0, 1)).permute(1, 2, 0)
    return torch.fft.irfft(out_ft, n=x.shape[-1])


class FNO1d(nn.Module):
  """
  1D Fourier Neural Operator.

  Maps a 1D function (represented on a grid with a positional coordinate)
  to a scalar output at each grid point. Uses three FNO layers, each combining
  a SpectralConv1D branch with a pointwise Conv1d residual branch.
  """

  def __init__(self,
               modes: int,
               width: int):
    """
    Initializes an FNO1d model.
    Args:
      modes: Number of Fourier modes to retain in each spectral layer.
      width: Hidden channel width used throughout the network.
    """
    super().__init__()
    self.linear_p = nn.Linear(2, width)
    self.spec1 = SpectralConv1D(width, width, modes)
    self.lin1 = nn.Conv1d(width, width, 1)
    self.spec2 = SpectralConv1D(width, width, modes)
    self.lin2 = nn.Conv1d(width, width, 1)
    self.spec3 = SpectralConv1D(width, width, modes)
    self.lin3 = nn.Conv1d(width, width, 1)
    self.linear_q = nn.Linear(width, 32)
    self.output_layer = nn.Linear(32, 1)
    self.activation = nn.Tanh()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the 1D FNO.
    Args:
      x: Input tensor of shape [B, N, 2] where the two channels are the
         function value and a positional coordinate.
    Returns:
      Output tensor of shape [B, N, 1].
    """
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
  """
  2D spectral convolution layer for use in FNO2d architectures.
  Applies learned linear mixing in the truncated 2D Fourier domain. The
  top-left (positive) and bottom-left (negative) frequency corners of the
  rfft2 spectrum are each multiplied by separate learned complex weight
  tensors, then transformed back to the spatial domain via irfft2.
  Weights are initialised with Glorot (Xavier) scaling:
  scale = sqrt(2 / (in_channels + out_channels)).
  """

  def __init__(self,
               in_channels: int,
               out_channels: int,
               modes1: int,
               modes2: int):
    """
    Initializes a SpectralConv2d layer.
    Args:
      in_channels: Number of input feature channels.
      out_channels: Number of output feature channels.
      modes1: Number of Fourier modes to retain along the height dimension.
      modes2: Number of Fourier modes to retain along the width dimension.
              Must satisfy modes2 <= (W * (1 + padding_frac)) // 2 + 1 at
              runtime to avoid shape mismatches in the einsum.
    """
    super().__init__()
    self.modes1 = modes1
    self.modes2 = modes2
    self.out_channels = out_channels
    self.scale = (2.0 / (in_channels + out_channels))**0.5
    self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
    self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the 2D spectral convolution.
    Args:
      x: Input tensor of shape [B, C_in, H, W].
    Returns:
      Real-valued output tensor of shape [B, C_out, H, W].
    """
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
  """
  2D Fourier Neural Operator for ptychographic image reconstruction.
  Maps a 2D diffraction pattern (plus spatial coordinate channels) to
  per-pixel amplitude and phase predictions. Each FNO layer combines a
  SpectralConv2d branch (global frequency mixing) with a pointwise Conv2d
  residual branch, followed by InstanceNorm2d and GELU activation.
  The spatial domain is zero-padded by `padding_frac` before the spectral
  layers to reduce aliasing at boundaries, and the padding is removed
  afterwards. The output head produces amplitude via sigmoid (range [0, 1])
  and phase via tanh * π (range [−π, π]).
  """

  def __init__(self,
               modes1: int,
               modes2: int,
               width: int,
               num_layers: int,
               padding_frac: float=0.25,
               in_channels: int=1):
    """
    Initializes an FNO2d model.
    Args:
      modes1: Number of Fourier modes along the height dimension.
      modes2: Number of Fourier modes along the width dimension.
      width: Hidden channel width used throughout the network.
      num_layers: Number of FNO layers (spectral + pointwise conv pairs).
      padding_frac: Fraction of the spatial size to zero-pad before spectral
                    layers (e.g. 0.25 pads by 25%). Set to 0.0 to disable.
      in_channels: Number of input channels per grid point (default 1 for
                   the diffraction pattern; set to 3 when passing x/y
                   coordinate grids as additional channels).
    """
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
    self.output_layer = nn.Linear(128, 2)
    self.activation = nn.GELU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the 2D FNO.
    Args:
      x: Input tensor of shape [B, H, W, C] where C matches `in_channels`.
    Returns:
      Output tensor of shape [B, H, W, 2] where the last dimension contains
      amplitude (channel 0, range [0, 1]) and phase (channel 1, range [−π, π]).
    """
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
