import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# Import the PaDiM model
from model import PaDiM
from dataset import MVTecDataset

# Define image dimensions and hyperparameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
D_REDUCED = 100 # number of randomly-selected feature channels kept per patch (paper default for resnet18)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data specific configuration
target_object = 'wood' # Change this to the object you want to train on (e.g., 'transistor')

# Define directory for saving models
MODEL_SAVE_DIR = './saved_models/' + target_object
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Define postfix for model filename. No epoch suffix -- unlike the
# gradient-trained AE/VAE stages, PaDiM has no epochs: fitting is a single
# pass over the training set that produces one artifact.
model_filename_prefix = f'padim_{target_object}'

# Ensure CUDA is available for GPU inference, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations. Unlike the AE/VAE stages' from-scratch encoders,
# PaDiM scores patches with a frozen ImageNet backbone, so images are
# normalized with ImageNet statistics instead of being left in [0, 1].
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Create dataset and dataloader for good training images
base_dir = '/content/mvtec/' + target_object
good_image_dir = os.path.join(base_dir, 'train/good')
train_dataset = MVTecDataset(root_dir=good_image_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of good training images for {target_object}: {len(train_dataset)}")

# Instantiate the model and fit the per-patch Gaussians over the training set.
# There is no backprop here -- this single pass over the "good" images is the
# entire "training" step.
model = PaDiM(d_reduced=D_REDUCED, device=device)
print(f"Fitting PaDiM (backbone: resnet18, d_reduced={D_REDUCED}) on {len(train_dataset)} images...")
model.fit(train_loader, device)
print(f"Fitted patch grid: {model.grid_size[0]}x{model.grid_size[1]} positions")

model_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}.pth')
model.save(model_path)
print(f"Model saved to {model_path}")
