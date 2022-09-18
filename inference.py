import torch
import torchvision
from torch.utils.data import DataLoader
from model import Unet
from dataset import CarvanaDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

VAL_IMG_DIR = "data/val/images/"
VAL_MASK_DIR = "data/val/annotations/"
BATCH_SIZE = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cpu"
):
    if not os.path.exists(folder):
        os.mkdir(folder)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

model = Unet()

validation_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2()
    ]
)


val_ds = CarvanaDataset(img_dir= VAL_IMG_DIR,
    mask_dir=VAL_MASK_DIR,
    transform=validation_transforms,
)

validation_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
)



checkpoint = torch.load("my_checkpoint.pth.tar", map_location=torch.device('cpu'))


model.load_state_dict(checkpoint["state_dict"])

save_predictions_as_imgs(
    validation_loader, model, folder="saved_images/", device="cpu"
)

# print(model)