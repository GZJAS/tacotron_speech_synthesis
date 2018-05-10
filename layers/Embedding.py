"""Embedding Layer"""

import tensorflow as tf


def EmbeddingLayer(input_, vocab_size, embedding_dim):
    """Embedding layers
    """
    output_ = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                        embeddings_initializer=tf.truncated_normal_initializer(
                                            mean=0.0, stddev=0.01))(input_)

    return output_
