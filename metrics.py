import torch
import numpy as np
from torchmetrics.functional.image import structural_similarity_index_measure


def ptycho_fno_loss(pred: torch.tensor, 
                    target: torch.tensor):
  """
  Loss function for Ptycho Reconstruction that respects the phase's 
  periodic nature. Weighs the amplitude and phase losses relative to 
  each other.
  """
    pred_amp = pred[..., 0]
    pred_phi = pred[..., 1]
    target_amp = target[..., 0]
    target_phi = target[..., 1]
    loss_amp = torch.nn.functional.mse_loss(pred_amp, target_amp)
    loss_phase = torch.mean(1 - torch.cos(pred_phi - target_phi))
    return 1.0 * loss_amp + 1.0 * loss_phase


def ptycho_complex_mse(pred: torch.tensor, 
                       target: torch.tensor):
  """
  Loss function for Ptycho Reconstruction that reconstructs the input, 
  and calculates the loss in complex space.
  """
  pred_amp = pred[..., 0]
  pred_phi = pred[..., 1]
  target_amp = target[..., 0]
  target_phi = target[..., 1]
  p_real = pred_amp * torch.cos(pred_phi)
  p_imag = pred_amp * torch.sin(pred_phi)
  t_real = target_amp * torch.cos(target_phi)
  t_imag = target_amp * torch.sin(target_phi)
  loss = torch.mean((p_real - t_real)**2 + (p_imag - t_imag)**2)
  return loss


def FRC(true_image: torch.tensor, 
        pred_image: torch.tensor, 
        bin_width: int=1):
  """
  Calculates Fourier Ring Correlation between target and predictions.
  Args:
    true_image: Target image of shape [H, W]
    pred_image: Prediction of shape [H, W]
    bin_width: width of frequency binning.
  Returns:
    numpy array of correlation values per frequency bin
  """
  h, w = true_image.shape
  ftrue = torch.fft.fftshift(torch.fft.fft2(true_image))
  fpred = torch.fft.fftshift(torch.fft.fft2(pred_image))
  numerator = ftrue * torch.conj(fpred)
  ftrue_norm = torch.abs(ftrue)**2
  fpred_norm = torch.abs(fpred)**2
  y, x = torch.meshgrid(torch.arange(h) - h//2, torch.arange(w) - w//2, indexing='ij')
  r = torch.sqrt(x**2 + y**2).to(true_image.device)
  r_int = r.round().long()
  max_r = h//2
  frc = []
  for i in range(0, max_r, bin_width):
    mask = (r_int >= i) & (r_int < i + bin_width)
    if mask.sum() > 0:
      top = torch.real(numerator[mask].sum())
      bottom = torch.sqrt(ftrue_norm[mask].sum() + fpred_norm[mask].sum())
      frc.append((top / (bottom + 1e-8)).item())
  return np.array(frc)


def SSIM(true_image: torch.Tensor, 
         pred_image: torch.Tensor, 
         data_range: float | tuple = 1.0)->float:
    """
    Computes SSIM between torch tensors, with shapes 
    [Batch, Channel, Height, Width] or [Channel, Height, Width].
    Args:
        true_image: Target images of shape [C, H, W]/[B, C, H, W].
        pred_image: Prediction of shape [C, H, W]/[B, C, H, W].
        data_range: Range of data (tuple or float).
    Returns:
        Float value of the SSIM.
    """
    if true_image.ndim == 3:
        true_image = true_image.unsqueeze(0)
    if pred_image.ndim == 3:
        pred_image = pred_image.unsqueeze(0)
    return ssim_fn(pred_image, true_image, data_range=data_range).detach().item()

