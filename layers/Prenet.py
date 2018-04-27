"""Prenet"""

import tensorflow as tf


class Prenet(tf.keras.Model):
    """Prenet
    """
    def __init__(self, prenet_layers, dropout):
        """Prenet
        """
        super(Prenet, self).__init__()

        self.prenet1 = tf.keras.layers.Dense(units=prenet_layers[0], activation=tf.nn.relu)
        self.prenet2 = tf.keras.layers.Dense(units=prenet_layers[1], activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, input_, training=True):
        """Run the model
            Args:
                input_: 3-D tensor of shape [batch_size, timesteps, input_dim]
            Returns:
                output_: 3-D tensor of shape [batch_size, timesteps, prenet_layers[-1]]
        """
        output1 = self.dropout(self.prenet1(input_), training=training)
        output2 = self.dropout(self.prenet2(output1), training=training)

        return output2
