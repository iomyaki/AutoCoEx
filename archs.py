import torch.nn as nn


class MLAE(nn.Module):
    def __init__(self, input_size, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 5000),
            nn.ReLU(),
            nn.Linear(5000, 1000),
            nn.ReLU(),
            nn.Linear(1000, bottleneck),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 1000),
            nn.ReLU(),
            nn.Linear(1000, 5000),
            nn.ReLU(),
            nn.Linear(5000, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AE(nn.Module):
    def __init__(self, input_size, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, bottleneck),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AE_ELU(nn.Module):
    def __init__(self, input_size, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, bottleneck),
            nn.ELU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, input_size),
            nn.ELU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
