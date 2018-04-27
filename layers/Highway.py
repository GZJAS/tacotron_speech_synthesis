"""Highway layer"""

import tensorflow as tf


class HighwayLayer(tf.keras.Model):
    """Highway Layer
    """
    def __init__(self,  num_units):
        super(HighwayLayer, self).__init__()

        self.pre_highway = tf.keras.layers.Dense(units=num_units)
        self.H = tf.keras.layers.Dense(units=num_units, activation=tf.nn.relu, use_bias=True)
        self.T = tf.keras.layers.Dense(units=num_units, activation=tf.nn.sigmoid, use_bias=True,
                                       bias_initializer=tf.constant_initializer(-1.0))

    def call(self, input_):
        """Run the model
            Args:
                input_: 3-D tensor of shape [batch_size, timesteps, input_dim]
            Returns:
                output_: 3-D tensor of shape [batch_size, timesteps, num_units]
        """
        # up/down-sample if the dimensionality of input does not match highway layer size
        if input_.get_shape()[-1] != self.num_units:
            input_ = self.pre_highway(input_)

        H_out = self.H(input_)
        T_out = self.T(input_)

        output_ = H_out * T_out + input_ * (1.0 - T_out)

        return output_
