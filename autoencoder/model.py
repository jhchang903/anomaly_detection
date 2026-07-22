import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # (Batch, 32, 256, 256) - Changed to 1 input channel
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                  # (Batch, 32, 128, 128)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (Batch, 64, 128, 128)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                  # (Batch, 64, 64, 64)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# (Batch, 128, 64, 64)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)                   # (Batch, 128, 32, 32)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # (Batch, 64, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (Batch, 32, 128, 128)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),   # (Batch, 1, 256, 256)
            nn.Sigmoid() # Output pixels between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
