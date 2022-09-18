import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import Unet
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import torchvision
from utils import (
    save_predictions_as_imgs,
    save_checkpoint,
    load_checkpoint,
    check_accuracy,
    train_epoch,
    validation_epoch
)


# ::::::::::::::ToTensorV2
# Convert image and mask to torch.Tensor. The numpy HWC image is converted to pytorch CHW tensor.
#  If the image is in HW format (grayscale image), it will be converted to pytorch HW tensor


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 300
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
TRAIN_IMG_DIR = "data/train/images/"
TRAIN_MASK_DIR = "data/train/annotations/"
VAL_IMG_DIR = "data/val/images/"
VAL_MASK_DIR = "data/val/annotations/"



model = Unet(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2()
    ]
)

validation_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2()
    ]
)

        

train_ds = CarvanaDataset(img_dir= TRAIN_IMG_DIR,
    mask_dir=TRAIN_MASK_DIR,
    transform=train_transforms,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=True,
)

val_ds = CarvanaDataset(img_dir= VAL_IMG_DIR,
    mask_dir=VAL_MASK_DIR,
    transform=validation_transforms,
)

validation_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=False,
)




scaler = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None 

for epoch in range(1, NUM_EPOCHS+1):
    train_epoch(train_loader, model, optimizer, loss_fn, epoch, DEVICE, scaler = scaler)
    validation_epoch(validation_loader, model, loss_fn=loss_fn, epoch=epoch, DEVICE=DEVICE)