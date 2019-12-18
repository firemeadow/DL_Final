import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib as plt
from gen import Generator
from disc import Discriminator


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


def train_step(data, label, gen_model, disc_model):
    pos_weight = torch.ones((1,)) * 0.99
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generated_price = gen_model(data).view(1, 1, 1)
    label = label.view(1, 1, 1)
    print(generated_price)
    fake_output = disc_model(generated_price)
    real_output = disc_model(label)

    gen_loss = generator_loss(fake_output, loss)
    disc_loss = discriminator_loss(real_output, fake_output, loss)
    return gen_loss, disc_loss, gen_model, disc_model


def train(data, num_attr, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, NUM_BATCHES, prices):
    gen_model = Generator(num_attr)
    gen_opt = optim.SGD(gen_model.parameters(), lr=LEARNING_RATE)
    disc_model = Discriminator()
    disc_opt = optim.SGD(disc_model.parameters(), lr=LEARNING_RATE)
    gens = []
    discs = []
    for epoch in range(NUM_EPOCHS):
        for i in range(NUM_BATCHES):
            gen_opt.zero_grad()  # zero the gradient buffers
            disc_opt.zero_grad()
            data_batch = data[i:i+BATCH_SIZE]
            label = prices[0][i+BATCH_SIZE]
            gen_loss, disc_loss, gen_model, disc_model = train_step(data_batch, label, gen_model, disc_model)
            if epoch is not NUM_EPOCHS-1:
                gen_loss.backward(retain_graph=True)
                disc_loss.backward(retain_graph=True)
            else:
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
    data = pd.read_csv('data.txt', delimiter=',', header=None)
    num_days = len(data) / 5
    train_days = int(np.ceil(num_days * (3 / 5)))
    num_attr = len(data.columns)
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 100
    BATCH_SIZE = 50
    NUM_BATCHES = train_days - BATCH_SIZE
    prices = torch.from_numpy(data[:train_days][4].values.astype(np.float32)).view(1, train_days)
    train_data = torch.from_numpy(data[:train_days].values.astype(np.float32))
    test_data = torch.from_numpy(data[train_days+1:].values.astype(np.float32))

    gens, discs, generator, discriminator = train(train_data, num_attr, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, NUM_BATCHES, prices)

