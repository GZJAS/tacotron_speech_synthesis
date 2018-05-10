"""CBHG module"""

from layers.Highway import HighwayNetwork
import tensorflow as tf


def BatchNormConv1D(input_, filters, kernel_size, strides, padding, activation=None):
    """1-D convolution with Batch normalization
    """
    conv1d_output = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                           activation=activation)(input_)
    batchnorm_output = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1d_output)

    return batchnorm_output


def Conv1DBankWithMaxPooling(input_, K, c, maxpool_width, maxpool_stride, activation=None):
    """1-D convolution bank with maxpooling along time axis
    """
    # conv 1-D banks
    bank_out = BatchNormConv1D(input_, filters=c, kernel_size=1, strides=1, padding="SAME", activation=activation)
    for k in range(2, K+1):
        output = BatchNormConv1D(bank_out, filters=c, kernel_size=k, strides=1, padding="SAME", activation=activation)
        bank_out = tf.concat((bank_out, output), axis=-1)

    # sanity check
    assert bank_out.get_shape()[-1] == K * c

    # maxpooling
    pool_out = tf.keras.layers.MaxPool1D(pool_size=maxpool_width, strides=maxpool_stride, padding="SAME")(bank_out)

    return pool_out


def CBHG(input_, K, c, maxpool_width, maxpool_stride, projections, num_highway, highway_size, gru_size):
    """CBHG module:The module is composed of:
         - 1-D convolution banks
         - Highway networks + residual connections
         - Bidirectional gated recurrent units
    """
    # 1-D conv bank with maxpooling
    convbank_output = Conv1DBankWithMaxPooling(input_, K, c, maxpool_width, maxpool_stride, activation=tf.nn.relu)

    # 1-D conv projections
    projection1_output = BatchNormConv1D(convbank_output, filters=projections[0], kernel_size=3, strides=1,
                                         padding="SAME", activation=tf.nn.relu)
    projection2_output = BatchNormConv1D(projection1_output, filters=projections[1], kernel_size=3, strides=1,
                                         padding="SAME")

    # residual connections
    projection2_output += tf.keras.layers.Dense(units=projections[1])(input_)

    # highway layers
    highway_output = HighwayNetwork(projection2_output, num_units=highway_size, num_layers=num_highway)

    # bidirectional GRU
    gru_output = tf.keras.layers.Bidirectional(
                                            tf.keras.layers.GRU(units=gru_size, return_sequences=True))(highway_output)

    return gru_output
