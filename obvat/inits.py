import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    return tf.get_variable(initializer=tf.random_uniform_initializer(), name=name, shape=shape)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    return tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), name=name, shape=shape)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.get_variable(initializer=tf.constant_initializer(initial), name=name, shape=shape)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.get_variable(initializer=tf.constant_initializer(initial), name=name, shape=shape)
