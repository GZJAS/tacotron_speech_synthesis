"""Experiment Hyperparameters"""

import tensorflow as tf


Hparams = tf.contrib.training.HParams(
    # text processing
    embedding_dim=256,

    # audio processing
    sampling_rate=16000,
    n_fft=2048,
    n_mels=80,
    frame_length=0.05,
    frame_shift=0.0125,
    preemphasis=0.97,
    max_db_level=100,
    ref_db_level=20,

    # griffin-lim
    power=1.2,
    num_griffin_lim_iters=50,

    # encoder prenet
    enc_prenet=[256, 128],
    enc_prenet_dropout=0.5,

    # decoder prenet
    dec_prenet=[256, 128],
    dec_prenet_dropout=0.5,

    # encoder CBHG
    enc_CBHG_K=16,
    enc_CBHG_c=128,
    enc_CBHG_maxpool_width=2,
    enc_CBHG_maxpool_stride=1,
    enc_CBHG_projections=[128, 128],
    enc_CBHG_num_highway=4,
    enc_CBHG_highway_size=128,
    enc_CBHG_gru_size=128,

    # attention decoder
    attn_gru_size=256,
    num_dec_gru_layers=2,
    dec_gru_size=256,

    # postnet CBHG
    post_CBHG_K=16,
    post_CBHG_c=128,
    post_CBHG_maxpool_width=2,
    post_CBHG_maxpool_stride=1,
    post_CBHG_projections=[256, 80],
    post_CBHG_num_highway=4,
    post_CBHG_highway_size=128,
    post_CBHG_gru_size=128,

    # model output
    r=5,

    # model training
    init_lr=0.001,
    batch_size=32,
)
