import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense

from utils.sparse_molecular_dataset import SparseMolecularDataset
from utils.utils import *

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class RelationalGCN(Layer) :
    def __init__(self, neurones, activation='relu') :
        super(RelationalGCN, self).__init__()
        self.neurones = neurones
        self.activation = activation
    
    # inputs : [X, A]
    def build(self, input_shape) :
        # shape of A
        adj_shape = input_shape[1]
        # number of features
        adj_num = adj_shape[3]
        # f_{y}^{l} eq (5) in the paper
        self.dense_links = [tf.keras.layers.Dense(self.neurones, activation='relu') for _ in range(adj_num)]
        self.dense_features = tf.keras.layers.Dense(self.neurones, activation='relu')
        
    # inputs : [X, A] 
    def call(self, inputs) :
        features = inputs[0]
        adj = inputs[1]
        adj = tf.transpose(adj, (0, 3, 1, 2))
        output = tf.stack([self.dense_links[i](features) for i in range(adj.shape[1])], 1)
        output = tf.matmul(adj, output)
        output = tf.reduce_sum(output, 1) + self.dense_features(features)
        return activations.get(self.activation)(output)


class Aggregation(Layer) :
    def __init__(self, neurones, activation='tanh') :
        super(Aggregation, self).__init__()
        self.neurones = neurones
        self.activation = activations.get(activation)

    def build(self, input_shape) :
        # sigma(i(h_{v}_{L} ...) in eq 6 in the paper
        self.dense_i = Dense(self.neurones, activation='sigmoid')
        # j(h_{v}^{L} in eq 6 in the paper
        self.dense_j = Dense(self.neurones, activation=self.activation)

    def call(self, inputs) :
        i = self.dense_i(inputs)
        j = self.dense_j(i)
        # h^{'}_{G} in eq 6 in the paper
        output = tf.reduce_sum(i*j, 1)
        output = self.activation(output)
        return output

class Discriminator(Model) :
    def __init__(self) :
        super(Model, self).__init__()
        self.gcl_1 = RelationalGCN(64, activation='tanh')
        self.gcl_2 = RelationalGCN(128, activation='tanh')

        self.aggregation = Aggregation(128, activation='tanh')

        self.dense1 = Dense(128, activation='tanh')
        self.dense2 = Dense(1, activation='tanh')

    def call(self, A, X) :
        x = self.gcl_1([X, A])
        x = self.gcl_2([x, A])

        x = self.aggregation(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

