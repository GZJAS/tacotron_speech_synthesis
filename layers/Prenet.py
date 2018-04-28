"""Prenet Module"""

import tensorflow as tf


class Prenet(tf.keras.Model):
    """Prenet module
    """
    def __init__(self, prenet_layers, dropout):
        """Constructor
        """
        super(Prenet, self).__init__()

        self.sampling_layer = tf.keras.layers.Dense(units=prenet_layers[0])
        self.prenet_layers = []
        for idx in range(len(prenet_layers)):
            self.prenet_layers.append(tf.keras.layers.Dense(units=prenet_layers[idx], activation=tf.nn.relu))
        self.prenet_dropout = tf.keras.layers.Dropout(dropout)

    def call(self, input_, training=True):
        """Run the model
            Args:
                 input_: 3-D tensor of shape [batch_size, timesteps, input_dim]
             Returns:
                 output_: 3-D tensor of shape [batch_size, timesteps, prenet_layers[-1]]
        """
        # output of sampling layer
        output_ = self.sampling_layer(input_)

        # output of prenet layers
        for idx in range(len(self.prenet_layers)):
            output_ = self.prenet_dropout(self.prenet_layers[idx](output_), training=training)

        return output_
