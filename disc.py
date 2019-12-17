import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.L1 = nn.Conv1d(1, 32, 1, stride=(2,))
        self.L2 = nn.LeakyReLU(0.01)
        self.L3 = nn.Conv1d(32, 64, 1, stride=(2,))
        self.L4 = nn.LeakyReLU(0.01)
        self.L5 = nn.Conv1d(64, 128, 1, stride=(2,))
        self.L6 = nn.LeakyReLU(0.01)
        self.L7 = nn.Linear(128, 220)
        self.L8 = nn.LeakyReLU(0.01)
        self.L9 = nn.Linear(220, 220)
        self.L10 = nn.ReLU()
        self.L11 = nn.Linear(220, 1)

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.L6(x)
        x = x.view(1, 128)
        x = self.L7(x)
        x = self.L8(x)
        x = self.L9(x)
        x = self.L10(x)
        x = self.L11(x)
        return x