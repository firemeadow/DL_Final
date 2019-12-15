import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib as plt
from gen import Generator
from disc import Discriminator
from load_data import load


def discriminator_loss(real_output, fake_output, loss):
    real_target = torch.ones(torch.np(real_output))
    fake_target = torch.zeros(torch.np(fake_output))
    real_loss = loss(real_output, real_target)
    fake_loss = loss(fake_output, fake_target)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, loss):
    fake_target = torch.ones(np.shape(fake_output))
    return loss(fake_output, fake_target)


def train_step(data, labels, gen_model, disc_model):
    num_attr = len(data.columns)
    pos_weights = torch.ones((BATCH_SIZE,))
    loss = nn.BCEWithLogitsLoss(pos_weights=pos_weights)
    noise = np.random.normal(size=(num_attr,))
    generated_samples = gen_model(noise)

    fake_output = disc_model(generated_samples)
    real_output = disc_model(data)

    gen_loss = generator_loss(fake_output, loss)
    disc_loss = discriminator_loss(real_output, fake_output, loss)
    return gen_loss, disc_loss


def train(data):
    learning_rate = 0.01
    NUM_EPOCHS = 100
    num_attr = len(data.columns)
    gen_model = Generator(num_attr)
    gen_opt = optim.SGD(gen_model.parameters(), lr=learning_rate)
    disc_model = Discriminator(num_attr)
    disc_opt = optim.SGD(disc_model.parameters(), lr=learning_rate)
    gens = []
    discs = []
    for epoch in range(NUM_EPOCHS):
        for data_batch in data:
            gen_opt.zero_grad()  # zero the gradient buffers
            disc_opt.zero_grad()
            gen_loss, disc_loss = train_step(data_batch, gen_model, disc_model)
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
    competitors = ['AVVIY', 'JEF', 'PGR', 'AIG', 'STFGX', 'BLK']
    data = load('BRK.A', 'BRK.B', competitors)
    num_days = len(data.columns) / 5
    BATCH_SIZE = 250
    train_days = int(np.ceil(num_days * (3 / 5)))
    NUM_BATCHES = int(np.ceil(train_days / BATCH_SIZE))
    #test_days = num_days - train_days

    ###BATCHING DOES NOT WORK

    raw_train_data = np.reshape(data[:train_days].to_numpy(), (train_days, 5))
    #raw_test_data = np.reshape(data[train_days + 1:].to_numpy(), (test_days, 5))

    batched_train_data = []
    for batch in range(NUM_BATCHES):
        temp = raw_train_data[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]
        batched_train_data[batch] = temp

    gens, discs, generator, discriminator = train(batched_train_data)


