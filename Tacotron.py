"""Tacotron model"""

import tensorflow as tf

from Hparams import Hparams as hp
from layers.Attention import AttentionDecoder
from layers.CBHG import CBHG
from layers.Prenet import Prenet


def encoder(input_):
    """Encoder
        Args:
            input_: 3-D tensor of shape [batch_size, T_x, embedding_dim]
            returns: 3-D tensor of shape [batch_size, T_x, 2*enc_CBHG_gru_size]
    """
    # encoder prenet
    enc_prenet_out = Prenet(input_, prenet_layers=hp.enc_prenet, dropout=hp.enc_prenet_dropout)

    # encoder CBHG
    enc_output = CBHG(enc_prenet_out, K=hp.enc_CBHG_K, c=hp.enc_CBHG_c, maxpool_width=hp.enc_CBHG_maxpool_width,
                      maxpool_stride=hp.enc_CBHG_maxpool_stride, projections=hp.enc_CBHG_projections,
                      num_highway=hp.enc_CBHG_num_highway, highway_size=hp.enc_CBHG_highway_size,
                      gru_size=hp.enc_CBHG_gru_size)

    return enc_output


def decoder(input_, memory):
    """Attention based decoder
        Args:
            input_: 3-D tensor of shape [batch_size, T_y//r, n_mels*r]; shifted (reshaped) log melspectrogram
            memory: 3-D tensor of shape [batch_size, T_x, 2*enc_CBHG_gru_size]; encoder output
            returns: 3-D tensor of shape [batch_size, T_y//r, n_mels*r]; predicted (reshaped) log melspectrogram
    """
    # decoder prenet
    dec_prenet_out = Prenet(input_, prenet_layers=hp.dec_prenet, dropout=hp.dec_prenet_dropout)

    # attention decoder
    attn_dec_out, state = AttentionDecoder(dec_prenet_out, memory, attn_gru_size=hp.attn_gru_size)
    
    # alignments
    alignments = tf.transpose(state.alignment_history.stack(), [1, 2, 0])

    # residual decoder GRUs
    for _ in range(hp.num_dec_gru_layers):
        dec_out = tf.keras.layers.GRU(units=hp.dec_gru_size)(attn_dec_out)
        attn_dec_out = tf.keras.layers.Add()([attn_dec_out, dec_out])

    # final affine layer
    mel_hats = tf.keras.layers.Dense(units=hp.n_mels * hp.r)(attn_dec_out)

    return mel_hats, alignments


def postnet(input_):
    """Post processing network
        Args:
            input_: 3-D tensor of shape [batch_size, T_y//r, n_mels*r]; (reshaped) log melspectrogram
            returns: 3-D tensor of shape [batch_size, T_y, 1+n_fft//2]; predicted log magnitude spectrogram
    """
    # reshape log melspectrogram
    input_ = tf.reshape(input_, [input_.shape[0], -1, hp.n_mels])

    # post-net CBHG
    post_CBHG_out = CBHG(input_, K=hp.post_CBHG_K, c=hp.post_CBHG_c, maxpool_width=hp.post_CBHG_maxpool_width,
                         maxpool_stride=hp.post_CBHG_maxpool_stride, projections=hp.post_CBHG_projections,
                         num_highway=hp.post_CBHG_num_highway, highway_size=hp.post_CBHG_highway_size,
                         gru_size=hp.post_CBHG_gru_size)

    # final affine layer
    log_hats = tf.keras.layers.Dense(units=1+hp.n_fft//2)(post_CBHG_out)

    return log_hats
