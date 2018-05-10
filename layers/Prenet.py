"""Prenet Module"""

import tensorflow as tf


def Prenet(input_, prenet_layers, dropout):
    """Prenet
    """
    output_ = input_
    for layer_size in prenet_layers:
        output_ = tf.keras.layers.Dense(units=layer_size, activation=tf.nn.relu)(output_)
        output_ = tf.keras.layers.Dropout(dropout)(output_)

    return output_
