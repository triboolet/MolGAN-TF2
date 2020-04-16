import tensorflow as tf
import numpy as np

def discriminator_loss(real_logits, fake_logits, penalty) :
    real_loss = tf.reduce_mean(fake_logits)
    fake_loss = tf.reduce_mean(real_logits)

    grad_penalty = tf.reduce_mean((np.linalg.norm(penalty[0]) - 1)**2) + tf.reduce_mean((np.linalg.norm(penalty[1]) - 1)**2)
    grad_penalty = tf.cast(grad_penalty, tf.float32)

    alpha = 10

    total_loss = -fake_loss + real_loss + alpha*grad_penalty

    return total_loss

def generator_loss(fake_logits) :
    return -tf.reduce_mean(fake_logits)


