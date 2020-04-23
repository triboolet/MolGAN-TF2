import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras as keras
from tensorflow.keras.layers import Lambda
from gcn import Discriminator
from generator import Generator
import numpy as np
import loss
import matplotlib.pyplot as plt

from utils.utils import *

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try :
    os.mkdir('results')
except FileExistsError :
    pass

def plot_images(epoch) :
    """
    Get a random batch of generated images
    Computes the validity score and plots the molecules
    """
    z = np.random.randn(32, 32)
    fA, fX = generator(z)
    fake_logits = discriminator(fA[0], fX[0])
    fA, fX = np.argmax(fA[0], axis=-1), np.argmax(fX[0], axis=-1)
    mols = [matrices2mol(n_, e_, strict=True) for n_, e_ in zip(fX, fA)]
    images = mols2grid_image(mols, 3)
    valid = get_valid_scores(mols)
    name = "results/result_" + str(epoch) + "_" + str(valid) + ".bmp"
    images.save(name)
    return valid
    
# data
A, X = get_molecules('data/gdb9.sdf')

# random shuffle
indexes = np.random.choice(len(A), 10000, replace=False)
A = A[indexes]
X = X[indexes]

A = tf.one_hot(A, depth=4, axis=3)
X = tf.one_hot(X, depth=5, axis=2)

N = len(A)

dataset = tf.data.Dataset.from_tensor_slices((A, X)).shuffle(N).batch(32)

#model
discriminator = Discriminator()
generator = Generator()

# optimizers
generator_optimizer = keras.optimizers.Adam(lr=0.008)
disciminator_optimizer = keras.optimizers.Adam()

gen_train_loss = keras.metrics.Mean(name='train_loss')
disc_train_loss = keras.metrics.Mean(name='disc_train_loss')

def train_step(molecules_A, molecules_X) :
    """
    First generates molecules with the generator, then pass a batch of data and the generated molecules through the discriminator, then applies backpropagation 
    """
    z = np.random.randn(molecules_A.shape[0], 32)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape :
        generated_A, generated_X = generator(z)

        real_logits = discriminator(molecules_A, molecules_X)
        fake_logits = discriminator(generated_A[0], generated_X[0])
        
        # backpropagation
        # gradient penalty : WGAN, discriminator loss
        with gen_tape.stop_recording() :
            with disc_tape.stop_recording() :
                # cf. equation (3) in the paper : penalty on the gradient of the discriminator wrt a linear combination of real and generated data

                epsilon_adj = tf.random.uniform(tf.shape(molecules_A), 0.0, 1.0, dtype=molecules_A.dtype)
                epsilon_features = tf.random.uniform(tf.shape(molecules_X), 0.0, 1.0, dtype=generated_X.dtype)

                with tf.GradientTape() as penalty_tape :
                    m1 = epsilon_adj * molecules_A
                    m2 = (1 - epsilon_adj)*generated_A[0]
                    x_hat_adj = m1 + m2
                    x_hat_features = epsilon_features * molecules_X + (1 - epsilon_features) * generated_X[0]
                    penalty_tape.watch([x_hat_adj, x_hat_features])
                    disc_penalty = discriminator(x_hat_adj, x_hat_features)

                # get the gradient, again eq (3) in the paper
                grad_adj = penalty_tape.gradient(disc_penalty, [x_hat_adj, x_hat_features])
         


        disc_loss = loss.discriminator_loss(real_logits, fake_logits, grad_adj) 
        gen_loss = loss.generator_loss(fake_logits)


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    disciminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))

    gen_train_loss(gen_loss)
    disc_train_loss(disc_loss)
    return gen_train_loss, disc_train_loss

EPOCHS = 20

valids = []
gen_losses = []
disc_losses = []

for epoch in range(0, EPOCHS) :

    print("Epoch : ", epoch)
    if epoch > 0 :
        valids.append(plot_images(epoch))

    for molecules in dataset :
        losses = train_step(molecules[0], molecules[1])
        gen_losses.append(losses[0].result())
        disc_losses.append(losses[1].result())

plt.subplot(2, 2, 1)
plt.title("Generator loss")
plt.grid()
plt.plot(gen_losses, c='grey')

plt.subplot(2, 2, 2)
plt.title("Discriminator loss")
plt.grid()
plt.plot(disc_losses, c='grey')

plt.subplot(2, 1, 2)
plt.title("Validation rate")
plt.grid()
plt.plot(valids, c='grey')

plt.show()
