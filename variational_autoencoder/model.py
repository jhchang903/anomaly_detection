import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Same conv backbone as the plain autoencoder, plus one extra
        # downsampling block so the flattened bottleneck stays a manageable
        # size for the fully-connected mu/logvar heads.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (Batch, 32, 256, 256)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                    # (Batch, 32, 128, 128)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (Batch, 64, 128, 128)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                    # (Batch, 64, 64, 64)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (Batch, 128, 64, 64)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                    # (Batch, 128, 32, 32)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# (Batch, 256, 32, 32)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                    # (Batch, 256, 16, 16)
        )
        self.encoder_out_shape = (256, 16, 16)
        encoder_out_dim = 256 * 16 * 16

        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, encoder_out_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # (Batch, 128, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # (Batch, 64, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # (Batch, 32, 128, 128)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),    # (Batch, 1, 256, 256)
            nn.Sigmoid() # Output pixels between 0 and 1
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, *self.encoder_out_shape)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon = self.decode(z)
        return recon, mu, logvar
