"""CBHG module"""

from layers.Highway import HighwayNetwork
import tensorflow as tf


class BatchNormConv1D(tf.keras.Model):
    """1D convolution with batch normalization
    """
    def __init__(self, filters, kernel_size, strides, padding, activation=None):
        """Constructor
        """
        super(BatchNormConv1D, self).__init__()

        # self.filters = filters              # number of output filters (dims of output space)
        # self.kernel_size = kernel_size      # length of 1-D convolution window
        # self.strides = strides              # stride length of the convolution
        # self.padding = padding              # one of "valid", "causal" or "same"
        # self.activation = activation        # activation function

        self.conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
                                             padding=padding, activation=activation)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)

    def call(self, input_):
        """Run the model
            Args:
                 input_: 3-D tensor of shape [batch_size, timesteps, input_dim]
             Returns:
                 output_: 3-D tensor of shape [batch_size, timesteps, filters]
        """
        return self.batch_norm(self.conv1d(input_))


class Conv1DBankWithMaxPooling(tf.keras.Model):
    """1D convolution bank with maxpooling along time axis
    """
    def __init__(self, K, c, maxpool_width, maxpool_stride, activation=None):
        """Constructor
        """
        super(Conv1DBankWithMaxPooling, self).__init__()

        self.K = K
        self.c = c

        # sampling layer
        self.sampling_layer = tf.keras.layers.Dense(units=c)

        # 1-D conv bank
        self.conv1D_bank = [BatchNormConv1D(filters=c, kernel_size=k, strides=1, padding="SAME", activation=activation)
                            for k in range(1, K+1)]

        # maxpooling
        self.maxpool = tf.keras.layers.MaxPool1D(pool_size=maxpool_width, strides=maxpool_stride, padding="SAME")

    def call(self, input_):
        """Run the model
            Args:
                 input_: 3-D tensor of shape [batch_size, timesteps, input_dim]
             Returns:
                 output_: 3-D tensor of shape [batch_size, timesteps, filters*K]
        """
        # output of sampling layer
        output_ = self.sampling_layer(input_)

        # output of Conv 1D bank
        output_ = tf.concat([conv1D(output_) for conv1D in self.conv1D_bank], axis=-1)

        # output of maxpooling
        output_ = self.maxpool(output_)

        return output_


class CBHG(tf.keras.Model):
    """CBHG module:The module is composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """
    def __init__(self, K, c, maxpool_width, maxpool_stride, projections, num_highway, highway_size, gru_size):
        """Constructor
        """
        super(CBHG, self).__init__()

        # 1-D conv bank with maxpooling
        self.convbank = Conv1DBankWithMaxPooling(K=K, c=c, maxpool_width=maxpool_width, maxpool_stride=maxpool_stride,
                                                 activation=tf.nn.relu)

        # 1-D conv projections
        self.conv_projection_1 = BatchNormConv1D(filters=projections[0], kernel_size=3, strides=1, padding="SAME",
                                                 activation=tf.nn.relu)
        self.conv_projection_2 = BatchNormConv1D(filters=projections[1], kernel_size=3, strides=1, padding="SAME")

        # residual sampling
        self.residual_sampling = tf.keras.layers.Dense(units=projections[1])

        # highway network
        self.highway_network = HighwayNetwork(num_units=highway_size, num_layers=num_highway)

        # bidirectional GRU
        self.bidirectional_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                                                               units=gru_size, return_sequences=True))

    def call(self, input_):
        """Run the model
            Args:
                 input_: 3-D tensor of shape [batch_size, timesteps, input_dim]
             Returns:
                 output_: 3-D tensor of shape [batch_size, timesteps, 2*gru_size]
        """
        # output of conv 1-D bank with maxpooling
        convbank_output = self.convbank(input_)

        # output of conv projections
        conv_projection_output = self.conv_projection_2(self.conv_projection_1(convbank_output))

        # residual connections

        conv_projection_output += self.residual_sampling(input_)

        # output of highway network
        highway_output = self.highway_network(conv_projection_output)

        # output of bidirectional GRU
        output_ = self.bidirectional_gru(highway_output)

        return output_
