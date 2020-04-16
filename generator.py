import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Reshape, Lambda, Dropout

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Dimensions
D = 32
batch_size = 10

N = 9
T = 5
Y = 4

class Generator(Model) :
    """
    Cf. section 3.1 and 5 ("Generator architecure") of the paper
    - input of 32 neurons, 3 layer MLP (128, 256, 512) with tanh as activation functions
    - projection of the output into the shapes of A (N, N, Y) and X (N, T) + gumbel softmax  
    """
    def __init__(self) :
        super(Generator, self).__init__()

        self.dropout_rate = .3
        self.dense1 = Dense(128, activation='tanh')
        self.dropout1 = Dropout(self.dropout_rate)
        self.dense2 = Dense(256, activation='tanh')
        self.dropout2 = Dropout(self.dropout_rate)
        self.dense3 = Dense(512, activation='tanh')
        self.dropout3 = Dropout(self.dropout_rate)

        self.denseA = Dense(N*N*Y, activation='relu')
        self.dropoutA = Dropout(self.dropout_rate)
        self.reshapeA = Reshape((N,N,Y))

        self.denseX = Dense(N*T)
        self.dropoutX = Dropout(self.dropout_rate)
        self.reshapeX = Reshape((N,T))

        self.argmax = Lambda(lambda x: tf.keras.backend.argmax(x))
        

    def call(self, z) :
        # MLP
        x = self.dense1(z)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.dropout3(x)

        # Projection
        A = self.denseA(x)
        A = self.dropoutA(A)
        A = self.reshapeA(A)

        X = self.denseX(x)
        X = self.dropoutX(X)
        X = self.reshapeX(X)

        # Gumbel softmax
        gumbel_dist_A = tfp.distributions.RelaxedOneHotCategorical(.15, A)
        generated_A = gumbel_dist_A.sample(1)

        gumbel_dist_X = tfp.distributions.RelaxedOneHotCategorical(0.15, X)
        generated_X = gumbel_dist_X.sample(1)

        return generated_A, generated_X
