import torch.nn as nn


def Generator(num_attr):
    model = nn.Sequential(
        nn.LSTM(num_attr, 500, 1),
        nn.Linear(1)
    )
    return model