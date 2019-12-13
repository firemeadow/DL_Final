import pandas as pd
import numpy as np
import matplotlib as plt
from gen import Generator
from disc import Discriminator

BUFFER_SIZE = 1000
BATCH_SIZE = 250
NUM_BATCHES = int(np.ceil(1000/250))
NUM_EPOCHS = 100

def discriminator_loss(real_output, fake_output):
    real_loss = real_output
    fake_loss = fake_output
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return fake_output

def train_step(data, labels, gen_model, disc_model):
    noise = np.random.normal()
    generated_samples = gen_model(noise)

    fake_output = disc_model(generated_samples)
    real_output = disc_model(data)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

    return gen_loss, disc_loss


def train(data, labels):
    gen_model = Generator()
    disc_model = Discriminator()
    gens = []
    discs = []
    for epoch in range(NUM_EPOCHS):
        for data_batch, label_batch in zip(data, labels):
            gen_loss, disc_loss = train_step(data_batch, label_batch, gen_model, disc_model)

        gen = gen_loss.numpy().mean()
        disc = disc_loss.numpy().mean()

        gens.append(gen)
        discs.append(disc)

    return gens, discs, generator, discriminator


if __name__ == '__main__':
    data = pd.read_csv('data.txt', delimiter=',')
    labels = data[[-1]]

    gens, discs, generator, discriminator = train(data, labels)


