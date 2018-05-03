"""Embedding Layer"""

import tensorflow as tf


class EmbeddingLayer(tf.keras.Model):
    """Embedding Layer
    """
    def __init__(self, vocab_size, embedding_dim):
        """Constructor
        """
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

    def call(self, input_):
        """Run the model
            Args:
                 input_: 2-D tensor of shape [batch_size, timesteps]
             Returns:
                 output_: 3-D tensor of shape [batch_size, timesteps, embedding_dim]
        """
        return self.embedding(input_)