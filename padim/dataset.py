import os
from torch.utils.data import Dataset
from PIL import Image


class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # PaDiM scores patches with a frozen ImageNet backbone, so images stay
        # RGB here (unlike the autoencoder/vae stages, which train their own
        # encoder from scratch and convert to grayscale).
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image
