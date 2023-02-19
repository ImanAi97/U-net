import numpy as np 
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import DataLoader

class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform = None):
        super(CarvanaDataset, self).__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # imgs are not full-path
        self.imgs = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        mask_path = os.path.join(self.mask_dir, self.imgs[index].replace("jpg", "gif"))
        img = Image.open(img_path).convert('RGB') ## RGB is th default
        img = np.array(img, dtype=np.float32) ## PIL image to numpy (for albumentation Data aumentation)
        ## array is uint8
        mask = Image.open(mask_path).convert('L') ## RGB is th default
        mask = np.array(mask, dtype=np.float32) ## PIL image to numpy (for albumentation Data aumentation)
        # mask array values are 0.0 or 255.0
        mask[mask == 255.0] = 1.0
        # each mask values is 0.0 or 1.0 and each pixel with 1.0 values belong to class 1 of segmentation        
        
        if self.transform is not None:
            augmentations = self.transform(image = img, mask = mask)
            img = augmentations["image"]
            mask = augmentations["mask"]
        
        # print(img_path)

        ## img, mask are in form of height, width, channels for mask height, width (2d)
        return img, mask


if __name__ =="__main__":
    ds = CarvanaDataset("data/val/images", "data/val/annotations")
    print(len(ds))
    img, mask = ds[1]
    print(img.shape)
    print(mask.shape)
    dl = DataLoader(ds, batch_size=32)
    img, mask = next(iter(dl))
    print(img)
    print(img.shape)
    print(mask.shape)
    print(len(dl))
