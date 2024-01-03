from torch import nn


BasicCNN = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
    # [B, 1, 28, 28] -> [B, 32, 28, 28]
    nn.MaxPool2d(kernel_size=2),
    # [B, 32, 28, 28] -> [B, 32, 14, 14]
    nn.LeakyReLU(),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    # [B, 32, 14, 14] -> [B, 64, 14, 14]
    nn.MaxPool2d(kernel_size=2),
    # [B, 64, 14, 14] -> [B, 64, 7, 7]
    nn.LeakyReLU(),

    nn.Flatten(), 
    # [B, 64, 7, 7] -> [B, 64 * 7 * 7]
    nn.Linear(64 * 7 * 7, 10),
)