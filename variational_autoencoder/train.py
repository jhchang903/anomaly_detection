import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import SSIM

# Import the VAE model
from model import VAE
from dataset import MVTecDataset

# Define image dimensions and hyperparameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0002
LOSS_FUNCTION_TYPE = 'l2' # 'l2' for MSE, 'ssim' for SSIM-based loss
LATENT_DIM = 128
# Weight on the KL divergence term (standard ELBO = 1.0). Lower this if
# reconstructions come out blurry/over-regularized -- a well-known VAE
# reconstruction-vs-regularization tradeoff, tune as needed per dataset.
KL_WEIGHT = 1.0

# Data specific configuration
target_object = 'wood' # Change this to the object you want to train on (e.g., 'transistor')

# Define directory for saving models
MODEL_SAVE_DIR = './saved_models/' + target_object
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Define postfix for model filename
model_filename_prefix = f'vae_{target_object}_{LOSS_FUNCTION_TYPE}'

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
model = VAE(latent_dim=LATENT_DIM).to(device)
print(model)

# Reconstruction loss function
if LOSS_FUNCTION_TYPE == 'l2':
    # Sum over pixels, mean over batch -- keeps the reconstruction term on
    # the same per-sample scale as the summed KL divergence term below.
    recon_criterion = nn.MSELoss(reduction='sum')
    loss_ylabel = 'Loss'
    loss_title_suffix = ' (MSE + KL)'
elif LOSS_FUNCTION_TYPE == 'ssim':
    recon_criterion = SSIM(data_range=1.0, size_average=True, channel=1) # channel=1 for grayscale
    loss_ylabel = 'Loss'
    loss_title_suffix = ' (SSIM + KL)'
else:
    raise ValueError(f"Unknown LOSS_FUNCTION_TYPE: {LOSS_FUNCTION_TYPE}")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
training_losses = []
reconstruction_losses = []
kl_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs = data.to(device)

        optimizer.zero_grad()
        outputs, mu, logvar = model(inputs)

        if LOSS_FUNCTION_TYPE == 'l2':
            recon_loss = recon_criterion(outputs, inputs) / inputs.size(0)
        elif LOSS_FUNCTION_TYPE == 'ssim':
            ssim_value = recon_criterion(outputs, inputs)
            recon_loss = 1 - ssim_value

        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = recon_loss + KL_WEIGHT * kl_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_recon_loss += recon_loss.item()
        running_kl_loss += kl_loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_recon_loss = running_recon_loss / len(train_loader)
    epoch_kl_loss = running_kl_loss / len(train_loader)
    training_losses.append(epoch_loss)
    reconstruction_losses.append(epoch_recon_loss)
    kl_losses.append(epoch_kl_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f} (Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f})")

    # Save model every 10 epochs
    if (epoch + 1) % 10 == 0:
        model_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

print("Training finished.")

# Plotting the training loss (total, reconstruction, and KL components)
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Total Loss')
plt.plot(reconstruction_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Divergence')
plt.title(f'PyTorch VAE Training Loss{loss_title_suffix}')
plt.xlabel('Epoch')
plt.ylabel(loss_ylabel)
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(MODEL_SAVE_DIR, f'{model_filename_prefix}_training_loss.png')
plt.savefig(loss_plot_path)
print(f"Saved training loss plot to {loss_plot_path}")
