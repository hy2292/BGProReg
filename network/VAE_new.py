import torch
from torch import nn
import torch.nn.functional as F


# Flatten layer
class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

# UnFlatten layer
class UnFlatten(nn.Module):
    def __init__(self, C, D, H, W):
        super(UnFlatten, self).__init__()
        self.C, self.D, self.H, self.W = C, D, H, W

    def forward(self, input):
        return input.reshape(input.size(0), self.C, self.D, self.H, self.W)

class VAE_new(nn.Module):
    def __init__(self, img_size=128, z_dim=32, nf=32):
        super(VAE_new, self).__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=nf, kernel_size=(3, 3, 3),stride=2, padding=1)
        self.bn1_enc = nn.BatchNorm3d(nf)

        self.conv2 = nn.Conv3d(in_channels=nf, out_channels=nf*2, kernel_size=(3, 3, 3), stride=2, padding=1)
        self.bn2_enc = nn.BatchNorm3d(nf*2)

        self.conv3 = nn.Conv3d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=(3, 3, 3), stride=2, padding=1)
        self.bn3_enc = nn.BatchNorm3d(nf * 4)

        self.conv4 = nn.Conv3d(in_channels=nf * 4, out_channels=nf * 8, kernel_size=(3, 3, 3), stride=2, padding=1)
        self.bn4_enc = nn.BatchNorm3d(nf * 8)


        h_dim = int(nf * 8 * img_size / 16 * img_size / 16 * img_size / 16)
        self.h_dim = h_dim
        self.dropout = nn.Dropout(0.25)
        self.fc11 = nn.Linear(h_dim, z_dim)
        self.fc12 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, h_dim)

        self.deconv1 = nn.ConvTranspose3d(in_channels=nf * 8 , out_channels=nf * 4, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1)
        self.bn1_dec = nn.BatchNorm3d(nf * 4)
        self.deconv2 = nn.ConvTranspose3d(in_channels=nf * 4 , out_channels=nf * 2, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1)
        self.deconv22 = nn.Conv3d(nf*2, nf*2, kernel_size=3, padding=1)
        self.bn2_dec = nn.BatchNorm3d(nf * 2)
        self.deconv3 = nn.ConvTranspose3d(in_channels=nf * 2 , out_channels=nf , kernel_size=(3, 3, 3), stride=2, padding=1,output_padding=1)
        self.deconv32 = nn.Conv3d(nf, nf, kernel_size=3, padding=1)
        self.bn3_dec = nn.BatchNorm3d(nf)
        self.deconv4 = nn.ConvTranspose3d(nf, nf, kernel_size=(3, 3, 3), stride=2, padding=1,output_padding=1)
        self.deconv42 = nn.Conv3d(nf, nf, kernel_size=3, padding=1)
        self.deconv21 = nn.Conv3d(nf*2, 3, kernel_size=3, padding=1)
        self.deconv31 = nn.Conv3d(nf, 3, kernel_size=3, padding=1)
        self.deconv41 = nn.Conv3d(nf, 3, kernel_size=3, padding=1)

        self.flatten = Flatten()
        self.unflatten = UnFlatten(C=int(nf * 8), D=int(img_size / 16),  H=int(img_size / 16), W=int(img_size / 16))

    def decoder(self, z):
        z = self.fc2(z)
        z = self.unflatten(z)
        z = F.relu(self.bn1_dec(self.deconv1(z)))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.bn2_dec(self.deconv22(z)))
        out1 = torch.tanh(self.deconv21(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.bn3_dec(self.deconv32(z)))
        out2 = torch.tanh(self.deconv31(z))
        out2 = out2 + F.interpolate(out1, size=(out2.shape[2],out2.shape[3],out2.shape[4]), mode='trilinear')
        z = F.relu(self.deconv4(z))
        z = F.relu(self.deconv42(z))
        out3 = torch.tanh(self.deconv41(z))
        out3 = out3 + F.interpolate(out2, size=(out3.shape[2], out3.shape[3], out3.shape[4]), mode='trilinear')
        return out3

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        mu= self.fc11(h)
        logvar = self.fc12(h)
        mu = mu.contiguous().view(-1, self.z_dim)
        logvar = logvar.contiguous().view(-1, self.z_dim)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = F.relu(self.bn1_enc(self.conv1(x)))
        h = F.relu(self.bn2_enc(self.conv2(h)))
        h = F.relu(self.bn3_enc(self.conv3(h)))
        h = F.relu(self.bn4_enc(self.conv4(h)))
        h = self.dropout(self.flatten(h))
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, max_norm):
        x = x / max_norm
        z, mu, logvar = self.encode(x)
        out = self.decode(z)
        return out * max_norm, mu, logvar


if __name__ == '__main__':
    x = torch.randn(1, 3, 128, 128, 128)
    VAE_model = VAE_new(img_size=128, z_dim=32, nf=32)
    VAE_model.train()
    max_norm = 3
    recon, mu, logvar = VAE_model(x, max_norm)