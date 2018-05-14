"""Attention Decoder"""

import tensorflow as tf


def AttentionDecoder(input_, memory, attn_gru_size):
    """Applies a GRU to input_, while attending memory
            input_: Decoder inputs
            memory: Encoder outputs
    """
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attn_gru_size, memory)
    decoder_cell = tf.contrib.rnn.GRUCell(attn_gru_size)
    cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attn_gru_size,
                                                              alignment_history=True)

    output_, state = tf.nn.dynamic_rnn(cell_with_attention, input_, dtype=tf.float32)

    return output_, state
