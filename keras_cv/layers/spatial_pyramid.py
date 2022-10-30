# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any
from typing import List
from typing import Mapping

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class SpatialPyramidPooling(tf.keras.layers.Layer):
    """Implements the Atrous Spatial Pyramid Pooling.

    References:
        [Rethinking Atrous Convolution for Semantic Image Segmentation](
          https://arxiv.org/pdf/1706.05587.pdf)
    [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
        Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

    inp = tf.keras.layers.Input((384, 384, 3))
    backbone = tf.keras.applications.EfficientNetB0(input_tensor=inp, include_top=False)
    layer_names = ['block2b_add', 'block3b_add', 'block5c_add', 'top_activation']

    backbone_outputs = {}
    for i, layer_name in enumerate(layer_names):
        backbone_outputs[i+2] = backbone.get_layer(layer_name).output

    # output_dict is a dict with 4 as keys, since it only process the level 4 backbone
    # inputs
    output_dict = keras_cv.layers.SpatialPyramidPooling(
        level=4, dilation_rates=[6, 12, 18])(backbone_outputs)

    # output[4].shape = [None, 16, 16, 256]
    """

    def __init__(
        self,
        level: int,
        dilation_rates: List[int],
        num_channels: int = 256,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs,
    ):
        """Initializes an Atrous Spatial Pyramid Pooling layer.

        Args:
            level: An `int` level to apply spatial pyramid pooling. This will be used to
                get the exact input tensor from the input dict in `call()`.
            dilation_rates: A `list` of integers for parallel dilated conv. Usually a
                sample choice of rates are [6, 12, 18].
            num_channels: An `int` number of output channels. Default to 256.
            activation: A `str` activation to be used. Default to 'relu'.
            dropout: A `float` for the dropout rate of the final projection output after
                the activations and batch norm. Default to 0.0, which means no dropout is
                applied to the output.
            **kwargs: Additional keyword arguments to be passed.
        """
        super().__init__(**kwargs)
        self.level = level
        self.dilation_rates = dilation_rates
        self.num_channels = num_channels
        self.activation = activation
        self.dropout = dropout

    def build(self, input_shape):
        # Retrieve the input at the level so that we can get the exact shape.
        if not isinstance(input_shape, dict):
            # raise ValueError(
            #     "SpatialPyramidPooling expects input features to be a dict with int keys, "
            #     f"received {input_shape}"
            # )
            input_shape_at_level = input_shape
        else:
            if self.level not in input_shape:
                raise ValueError(
                    f"SpatialPyramidPooling expect the input dict to contain key {self.level}, "
                    f"received {input_shape}"
                )
            input_shape_at_level = input_shape[self.level]

        height = input_shape_at_level[1]
        width = input_shape_at_level[2]
        channels = input_shape_at_level[3]

        # This is the parallel networks that process the input features with different
        # dilation rates. The output from each channel will be merged together and feed
        # to the output.
        self.aspp_parallel_channels = []

        # Channel1 with Conv2D and 1x1 kernel size.
        conv_sequential = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.num_channels, kernel_size=(1, 1), use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                ),
                tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
                tf.keras.layers.Activation(self.activation),
            ],
            name="squential_conv1",
        )
        self.aspp_parallel_channels.append(conv_sequential)

        # Channel 2 and afterwards are based on self.dilation_rates, and each of them
        # will have conv2D with 3x3 kernel size.
        for dilation_rate in self.dilation_rates:
            conv_sequential = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        filters=self.num_channels,
                        kernel_size=(3, 3),
                        padding="same",
                        dilation_rate=dilation_rate,
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                        use_bias=False,
                    ),
                    tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
                    tf.keras.layers.Activation(self.activation),
                ],
                name="sequential_dilation_{}".format(dilation_rate)
            )
            self.aspp_parallel_channels.append(conv_sequential)
        
        
        # Last channel is the global average pooling with conv2D 1x1 kernel.
        pool_sequential = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(name='seq_pool_gap'),
                tf.keras.layers.Reshape((1, 1, channels), name='seq_pool_res'),
                tf.keras.layers.Conv2D(
                    filters=self.num_channels, kernel_size=(1, 1), use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    name='seq_pool_conv'
                ),
                tf.keras.layers.experimental.SyncBatchNormalization(),
                # tf.keras.layers.BatchNormalization(momentum=0.9997, epsilon=1.001e-5, name='seq_pool_bn', fused=True),
                tf.keras.layers.Activation(self.activation, name='seq_pool_act'),
                tf.keras.layers.Resizing(height, width, interpolation="bilinear", name='seq_pool_rez'),
                # tf.keras.layers.UpSampling2D(size=(height, width))
            ],
            name="sequential_pool",
        )
        self.pool_sequential = pool_sequential
        self.aspp_parallel_channels.append(pool_sequential)

        # Final projection layers
        self.projection = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.num_channels, kernel_size=(1, 1), use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                ),
                tf.keras.layers.BatchNormalization(epsilon=1.001e-5),
                tf.keras.layers.Activation(self.activation),
                tf.keras.layers.Dropout(rate=self.dropout),
            ],
            name="sequential_proj",
        )

    def call(self, inputs, training=None):
        """Calls the Atrous Spatial Pyramid Pooling layer on an input.

        The input the of the layer will be a dict of {`level`, `tf.Tensor`}, and layer
        will pick the actual input based on the `level` from its init args.
        The output of the layer will be a dict of {`level`, `tf.Tensor`} with only one level.

        Args:
          inputs: A `dict` of `tf.Tensor` where
            - key: A `int` of the level of the multilevel feature maps.
            - values: A `tf.Tensor` of shape [batch, height_l, width_l,
              filter_size].

        Returns:
          A `dict` of `tf.Tensor` where
            - key: A `int` of the level of the multilevel feature maps.
            - values: A `tf.Tensor` of output of SpatialPyramidPooling module. The shape
              of the output should be [batch, height_l, width_l, num_channels]
        """
        if not isinstance(inputs, dict):
            input_at_level = inputs
            # raise ValueError(
            #     "SpatialPyramidPooling expects input features to be a dict with int keys, "
            #     f"received {inputs}"
            # )
        else:
            if self.level not in inputs:
                raise ValueError(
                    f"SpatialPyramidPooling expect the input dict to contain key {self.level}, "
                    f"received {inputs}"
                )
            input_at_level = inputs[self.level]
        # tf.debugging.assert_all_finite(input_at_level, "aspp input inf")
        result = []
        for channel in self.aspp_parallel_channels:
            temp = tf.cast(
                channel(input_at_level, training=training), input_at_level.dtype
            )
            # tf.debugging.assert_all_finite(temp, "channel output inf {}".format(channel.name))
            result.append(temp)

        # temp = input_at_level
        # for l in self.pool_sequential.layers:
        #     tf.print(tf.reduce_min(temp), "layer {} input min".format(l.name))
        #     tf.print(tf.reduce_max(temp), "layer {} input max".format(l.name))
        #     for var in l.variables:
        #         tf.print(tf.reduce_min(var), "layer var {} min".format(var.name))
        #         tf.print(tf.reduce_max(var), "layer var {} max".format(var.name))
        #     temp = l(temp, training=training)
        #     tf.debugging.assert_all_finite(temp, "layer {} inf".format(l.name))

        # for var in self.pool_sequential.layers[3].variables:
        #     if "moving_variance" in var.name:
        #         tf.debugging.assert_all_finite(var, "moving_variance inf {}".format(var.name))

        # if tf.random.uniform([], minval=0., maxval=1.) > 0.99:
        #     for var in self.pool_sequential.layers[3].variables:
        #         if "moving_variance" in var.name:
        #             tf.print(tf.reduce_max(var), "var {} max".format(var.name))
        #             tf.print(tf.reduce_min(var), "var {} min".format(var.name))

        result = tf.concat(result, axis=-1)
        result = self.projection(result, training=training)
        # tf.debugging.assert_all_finite(result, "aspp proj inf {}".format(channel.name))
        if not isinstance(inputs, dict):
            return result
        return {self.level: result}

    def get_config(self) -> Mapping[str, Any]:
        config = {
            "level": self.level,
            "dilation_rates": self.dilation_rates,
            "num_channels": self.num_channels,
            "activation": self.activation,
            "dropout": self.dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
