import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib as plt
from gen import Generator
from disc import Discriminator

BUFFER_SIZE = 1000
BATCH_SIZE = 250
NUM_BATCHES = int(np.ceil(BUFFER_SIZE/BATCH_SIZE))
NUM_EPOCHS = 100
pos_weights = torch.ones((BATCH_SIZE,))
loss = nn.BCEWithLogitsLoss(pos_weights=pos_weights)
learning_rate = 0.01

def discriminator_loss(real_output, fake_output):
    real_target = torch.ones(torch.shape(real_output))
    fake_target = torch.zeros(torch.shape(fake_output))
    real_loss = loss(real_output, real_target)
    fake_loss = loss(fake_output, fake_target)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    fake_target = torch.ones(torch.shape(fake_output))
    return loss(fake_output, fake_target)


def train_step(data, labels, gen_model, disc_model):
    num_attr = len(data.columns)
    noise = np.random.normal(size=(num_attr,))
    generated_samples = gen_model(noise)

    fake_output = disc_model(generated_samples)
    real_output = disc_model(data)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    return gen_loss, disc_loss


def train(data, labels):
    num_attr = len(data.columns)
    gen_model = Generator(num_attr)
    gen_opt = optim.SGD(gen_model.parameters(), lr=learning_rate)
    disc_model = Discriminator(num_attr)
    disc_opt = optim.SGD(disc_model.parameters(), lr=learning_rate)
    gens = []
    discs = []
    for epoch in range(NUM_EPOCHS):
        for data_batch, label_batch in zip(data, labels):
            gen_opt.zero_grad()  # zero the gradient buffers
            disc_opt.zero_grad()
            gen_loss, disc_loss = train_step(data_batch, label_batch, gen_model, disc_model)
            gen_loss.backward()
            disc_loss.backward()
            gen_opt.step()
            disc_opt.step()

        gen = gen_loss.numpy().mean()
        disc = disc_loss.numpy().mean()

        gens.append(gen)
        discs.append(disc)

    return gens, discs, generator, discriminator


if __name__ == '__main__':
    data = pd.read_csv('data.txt', delimiter=',')
    labels = data[[-1]]

    gens, discs, generator, discriminator = train(data, labels)


