import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchmetrics
import os
import argparse
from data_setup import PtychoDataset
from models import FNO2d
from engine import train
from utils import get_devices, set_seeds, save_model, plot_results
from metrics import FRC, SSIM


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./gdrive/MyDrive/PtychoNNData/")
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--modes", type=int, default=16)
parser.add_argument("--width", type=int, default=32)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--model", type=str, default="PtychoNN")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

device = get_devices()
set_seeds(args.seed)
train_ds = PtychoDataset(args.data_dir+"X_train_final.npy", args.data_dir+"Y_I_train_final.npy", args.data_dir+"Y_P_train_final.npy")
val_ds = PtychoDataset(args.data_dir+"X_val_final.npy", args.data_dir+"Y_I_val_final.npy", args.data_dir+"Y_P_val_final.npy")
train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=args.batch_size)

device = get_devices()
model = FNO2d(args.modes, args.modes, args.width, args.num_layers, 1.0, 3, 2).to(device)
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
results = train(model, train_dl, val_dl, loss_fn, opt, device, 25, scheduler)
model_name = args.model + str(args.num_epochs)
save_model("./Models", "model_" + args.model, model)
plot_results(model, val_dl, device, n=6)

