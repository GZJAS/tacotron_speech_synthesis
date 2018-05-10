"""Highway Network"""

import tensorflow as tf


def HighwayNetwork(input_, num_units, num_layers):
    """Highway Network
    """
    output_ = tf.keras.layers.Dense(units=num_units)(input_)
    for idx in range(num_layers):
        # H
        transformed = tf.keras.layers.Dense(units=num_units, activation=tf.nn.relu)(input_)

        # T
        gate = tf.keras.layers.Dense(units=num_units, activation=tf.nn.sigmoid,
                                     bias_initializer=tf.keras.initializers.Constant(-1.0))(output_)
        
        # 1.0 - T
        negated_gate = tf.keras.layers.Lambda(lambda x: 1.0 - x, output_shape=(num_units, ))(gate)

        # H * T
        transform_gated = tf.keras.layers.Multiply()([gate, transformed])
        # input * (1.0 - T)
        identity_gated = tf.keras.layers.Multiply()([negated_gate, output_])

        # final output = H * T + input * (1.0 - T)
        output_ = tf.keras.layers.Add()([transform_gated, identity_gated])

    return output_
