import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import SSIM

# Import the Autoencoder model
from model import Autoencoder
from dataset import MVTecDataset

# Define image dimensions and hyperparameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0002
LOSS_FUNCTION_TYPE = 'l2' # 'l2' for MSE, 'ssim' for SSIM-based loss

# Data specific configuration
target_object = 'wood' # Change this to the object you want to train on (e.g., 'transistor')

# Define directory for saving models
MODEL_SAVE_DIR = './saved_models/' + target_object
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Define postfix for model filename
model_filename_prefix = f'autoencoder_{target_object}_{LOSS_FUNCTION_TYPE}'

# Ensure CUDA is available for GPU training, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(), # Converts PIL Image to PyTorch Tensor (H*W*C to C*H*W) and normalizes to [0, 1]
])

# Create dataset and dataloader for good training images
base_dir = '/content/mvtec/' + target_object
good_image_dir = os.path.join(base_dir, 'train/good')
train_dataset = MVTecDataset(root_dir=good_image_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Number of good training images for {target_object}: {len(train_dataset)}")

# Instantiate the model and move to device
model = Autoencoder().to(device)
print(model)

# Loss function and optimizer
if LOSS_FUNCTION_TYPE == 'l2':
    criterion = nn.MSELoss()
    loss_ylabel = 'MSE Loss'
    loss_title_suffix = ' (MSE)'
elif LOSS_FUNCTION_TYPE == 'ssim':
    criterion = SSIM(data_range=1.0, size_average=True, channel=1) # channel=1 for grayscale
    loss_ylabel = '1 - SSIM Loss'
    loss_title_suffix = ' (SSIM)'
else:
    raise ValueError(f"Unknown LOSS_FUNCTION_TYPE: {LOSS_FUNCTION_TYPE}")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
training_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs = data.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if LOSS_FUNCTION_TYPE == 'l2':
            loss = criterion(outputs, inputs)
        elif LOSS_FUNCTION_TYPE == 'ssim':
            ssim_value = criterion(outputs, inputs)
            loss = 1 - ssim_value
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    training_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    # Save model every 10 epochs
    if (epoch + 1) % 10 == 0:
        model_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

print("Training finished.")

# Plotting the training loss
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.title(f'PyTorch Autoencoder Training Loss{loss_title_suffix}')
plt.xlabel('Epoch')
plt.ylabel(loss_ylabel)
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_training_loss.png')
plt.savefig(loss_plot_path)
print(f"Saved training loss plot to {loss_plot_path}")
