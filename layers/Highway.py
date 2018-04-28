"""Highway Network"""

import tensorflow as tf


class HighwayNetwork(tf.keras.Model):
    """Highway network
    """
    def __init__(self, num_units, num_layers):
        """Constructor
        """
        super(HighwayNetwork, self).__init__()

        self.num_layers = num_layers
        self.num_units = num_units

        self.sampling_layer = tf.keras.layers.Dense(units=num_units)

        self.Hs = []
        self.Ts = []

        for idx in range(self.num_layers):
            self.Hs.append(tf.keras.layers.Dense(units=num_units, activation=tf.nn.relu))
            self.Ts.append(tf.keras.layers.Dense(units=num_units, activation=tf.nn.sigmoid,
                           bias_initializer=tf.constant_initializer(-1.0)))

    def call(self, input_):
        """ Run the model
            Args:
                 input_: 3-D tensor of shape [batch_size, timesteps, input_dim]
             Returns:
                 output_: 3-D tensor of shape [batch_size, timesteps, num_units]
        """
        # output of sampling layer
        output_ = self.sampling_layer(input_)

        # output of highway layers
        for idx in range(self.num_layers):
            H_out = self.Hs[idx](output_)
            T_out = self.Ts[idx](output_)

            output_ = H_out * T_out + output_ * (1.0 - T_out)

        return output_
