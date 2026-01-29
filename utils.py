import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt


def get_devices():
  return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seeds(seed: int=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


def save_model(model_dir: str, 
               model_name: str, 
               model: torch.nn.Module):
  """
  Saves pytorch model in model_dir with model_name.
  Args:
    model_dir: Directory to save model in.
    model_name: name of file to store model.
    model: model to be saved.
  Returns:
    None
  """
  os.makedirs(model_dir, exist_ok=True)
  if not model_name.endswith("pt"):
    model_name += ".pt"
  torch.save(model.state_dict(), os.path.join(model_dir, model_name))


def plot_results(model, 
                 test_dl, 
                 device, 
                 n: int=6, 
                 save: bool=True,
                 fig_path: str="temp"):
  model.eval()
  xb, yb = next(iter(test_dl))              #xb:[B, N, N, 3],yb:[B, N, N, 2]
  with torch.no_grad():
    preds = model(xb.to(device))            #preds: [B, N, N, 2]
  diff_patterns, true_amplitudes, true_phases = xb[:,:,:,0], yb[:,:,:,0], yb[:,:,:,1]
  pred_amplitudes, pred_phases = preds[:,:,:,0].cpu(), preds[:,:,:,1].cpu()
  fig, ax = plt.subplots(5, n, figsize=(10, 14))
  for i in range(n):
    im=ax[0,i].imshow(diff_patterns[i].numpy())
    plt.colorbar(im, ax=ax[0,i], format='%.2f')
    ax[0,i].get_xaxis().set_visible(False)
    ax[0,i].get_yaxis().set_visible(False)

    im=ax[1,i].imshow(true_amplitudes[i].numpy())
    plt.colorbar(im, ax=ax[1,i], format='%.2f')
    ax[1,i].get_xaxis().set_visible(False)
    ax[1,i].get_yaxis().set_visible(False)

    im=ax[2,i].imshow(pred_amplitudes[i].numpy())
    plt.colorbar(im, ax=ax[2,i], format='%.2f')
    ax[2,i].get_xaxis().set_visible(False)
    ax[2,i].get_yaxis().set_visible(False)

    im=ax[3,i].imshow(true_phases[i].numpy())
    plt.colorbar(im, ax=ax[3,i], format='%.2f')
    ax[3,i].get_xaxis().set_visible(False)
    ax[3,i].get_yaxis().set_visible(False)

    im=ax[4,i].imshow(pred_phases[i].numpy())
    plt.colorbar(im, ax=ax[4,i], format='%.2f')
    ax[4,i].get_xaxis().set_visible(False)
    ax[4,i].get_yaxis().set_visible(False)

  if save:
    path = fig_path + ".pdf"
    fig.savefig(path, bbox_inches='tight')
