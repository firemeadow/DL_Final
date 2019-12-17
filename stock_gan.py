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
    real_target = torch.ones(np.shape(real_output))
    fake_target = torch.zeros(np.shape(fake_output))
    real_loss = loss(real_output, real_target)
    fake_loss = loss(fake_output, fake_target)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, loss):
    fake_target = torch.ones(np.shape(fake_output))
    return loss(fake_output, fake_target)


def train_step(data, label, num_attr, gen_model, disc_model, BATCH_SIZE):
    pos_weight = torch.ones((BATCH_SIZE,))
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generated_samples = gen_model(data)
    prices = data[:][4]
    fake_output = disc_model(generated_samples)
    real_output = disc_model(label)

    gen_loss = generator_loss(fake_output, loss)
    disc_loss = discriminator_loss(real_output, fake_output, loss)
    return gen_loss, disc_loss, gen_model, disc_model


def train(data, num_attr, minibatch_size, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, NUM_BATCHES):
    gen_model = Generator(num_attr, minibatch_size)
    gen_opt = optim.SGD(gen_model.parameters(), lr=LEARNING_RATE)
    disc_model = Discriminator(num_attr)
    disc_opt = optim.SGD(disc_model.parameters(), lr=LEARNING_RATE)
    gens = []
    discs = []
    for epoch in range(NUM_EPOCHS):
        for i in range(NUM_BATCHES):
            gen_opt.zero_grad()  # zero the gradient buffers
            disc_opt.zero_grad()
            data_batch = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            label = data_batch[len(data_batch)-1][4]
            data_batch = data_batch[:len(data_batch)-1]
            gen_loss, disc_loss, gen_model, disc_model = train_step(data_batch, label, num_attr, gen_model, disc_model, BATCH_SIZE)
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
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 100
    data = pd.read_csv('data.txt', delimiter=',', header=None)
    num_days = len(data) / 5
    BATCH_SIZE = 250
    train_days = int(np.ceil(num_days * (3 / 5)))
    NUM_BATCHES = int(np.floor(train_days / BATCH_SIZE))
    num_attr = len(data.columns)
    train_data = torch.from_numpy(data[:train_days].values.astype(np.float32))
    test_data = torch.from_numpy(data[train_days+1:].values.astype(np.float32))
    minibatch_size = 50

    gens, discs, generator, discriminator = train(train_data, num_attr, minibatch_size, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, NUM_BATCHES)

