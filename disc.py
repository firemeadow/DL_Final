import torch.nn as nn


def Discriminator(num_attr):
    model = nn.Sequential(
        nn.Conv1d(num_attr, 32, (5,), stride=(2,)),
        nn.LeakyReLU(0.01),
        nn.Conv1d(32, 64, (5,), stride=(2,)),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(64),
        nn.Conv1d(64, 128, (5,), stride=(2,)),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(128),
        nn.Linear(128, 220),
        nn.BatchNorm1d(220),
        nn.LeakyReLU(0.01),
        nn.Linear(220, 220),
        nn.ReLU(),
        nn.Linear(220, 1)
    )
    return model