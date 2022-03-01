import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

latent_dim = 30

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation = 'relu', input_shape=[latent_dim]),
    tf.keras.layers.Dense(150, activation = 'relu'),
    tf.keras.layers.Dense(2, activation = 'linear')
    ])

discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=((2,))),
    tf.keras.layers.Dense(150, activation = 'relu'),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

gan = tf.keras.models.Sequential([generator, discriminator])

discriminator.compile(loss = 'binary_crossentropy', optimizer = 'Adam')
discriminator.trainable = False
gan.compile(loss = 'binary_crossentropy', optimizer = 'Adam')

batch_size = 5

datasetX1 = np.random.uniform(low = -0.5, high =0.5, size = 2000)
datasetX2 = [x**2 for x in datasetX1]
dataset = [(x,y) for x, y in zip(datasetX1, datasetX2)]

dataset = tf.data.Dataset.from_tensor_slices(dataset)
dataset = dataset.batch(batch_size, drop_remainder = True).prefetch(1)

def train_gan(gan, dataset, batch_size, codings_size, n_epochs = 1000):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            noise = tf.random.normal(shape=[100, latent_dim])
            test = generator(noise)
            plt.scatter(test[:,0], test[:,1])
            plt.show()
            plt.clf()
        for X_batch in dataset:
            #phase 1 
            X_batch = tf.cast(X_batch, tf.float32)
            noise = tf.random.normal(shape=[batch_size, latent_dim])
            generated_function = generator(noise)
            X_fake_and_real = tf.concat([generated_function, X_batch], axis = 0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            #phase 2
            noise = tf.random.normal(shape=[batch_size, latent_dim])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)



train_gan(gan,dataset,batch_size, latent_dim)

noise = tf.random.normal(shape=[60, latent_dim])
test = generator(noise)
plt.scatter(test[:,0], test[:,1])
plt.show()
plt.clf()
