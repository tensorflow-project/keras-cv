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

import tensorflow as tf
from tensorflow.keras import layers

import keras_cv

BN_AXIS = 3

def Block(filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """

    def apply(x):
        if conv_shortcut:
            shortcut = layers.Conv2D(
                4 * filters, 1, strides=stride, use_bias=False, name=name + "_0_conv"
            )(x)
            shortcut = layers.BatchNormalization(
                axis=BN_AXIS, epsilon=1.001e-5, name=name + "_0_bn"
            )(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(
            filters, 1, strides=stride, use_bias=False, name=name + "_1_conv"
        )(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_1_bn"
        )(x)
        x = layers.Activation("relu", name=name + "_1_relu")(x)

        x = layers.Conv2D(
            filters, kernel_size, padding="SAME", dilation_rate=2, use_bias=False, name=name + "_2_conv"
        )(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_2_bn"
        )(x)
        x = layers.Activation("relu", name=name + "_2_relu")(x)

        x = layers.Conv2D(4 * filters, 1, use_bias=False, name=name + "_3_conv")(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_3_bn"
        )(x)

        x = layers.Add(name=name + "_add")([shortcut, x])
        x = layers.Activation("relu", name=name + "_out")(x)
        return x

    return apply

def Stack(filters, blocks, stride=2, name=None, block_fn=Block, first_shortcut=True):
    """A set of stacked residual blocks.
    Args:
      filters: integer, filters of the layers in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
      block_fn: callable, `Block` or `BasicBlock`, the block function to stack.
      first_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
    Returns:
      Output tensor for the stacked blocks.
    """

    def apply(x):
        x = block_fn(
            filters, stride=stride, name=name + "_block1", conv_shortcut=first_shortcut
        )(x)
        for i in range(2, blocks + 1):
            x = block_fn(filters, conv_shortcut=False, name=name + "_block" + str(i))(x)
        return x

    return apply

class ResnetDeepLabV3(tf.keras.models.Model):
    """A segmentation model based on the DeepLab v3.

    Args:
        classes: int, the number of classes for the detection model. Note that
            the classes doesn't contain the background class, and the classes
            from the data should be represented by integers with range
            [0, classes).
        include_rescaling: boolean, whether to Rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        backbone: an optional backbone network for the model. Can be a `tf.keras.layers.Layer`
            instance. The supported pre-defined backbone models are:
            1. "resnet50_v2", a ResNet50 V2 model
            Default to 'resnet50_v2'.
        decoder: an optional decoder network for segmentation model, e.g. FPN. The
            supported premade decoder is: "fpn". The decoder is called on
            the output of the backbone network to up-sample the feature output.
            Default to 'fpn'.
        segmentation_head: an optional `tf.keras.Layer` that predict the segmentation
            mask based on feature from backbone and feature from decoder.

    """

    def __init__(
        self,
        classes,
        include_rescaling,
        backbone="resnet50_v2",
        segmentation_head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.classes = classes
        # ================== Backbone and weights. ==================
        if isinstance(backbone, str):
            supported_premade_backbone = [
                "resnet50_v2",
            ]
            if backbone not in supported_premade_backbone:
                raise ValueError(
                    "Supported premade backbones are: "
                    f'{supported_premade_backbone}, received "{backbone}"'
                )
            self._backbone_passed = backbone
            if backbone == "resnet50_v2":
                backbone = keras_cv.models.ResNet50(
                    include_rescaling=include_rescaling, include_top=False,
                    input_shape=[512, 512, 3]
                )
                inputs = tf.keras.Input(shape=[None, None, 3])
                outputs = backbone(inputs)
                # outputs = Stack(
                #     filters=512,
                #     blocks=3,
                #     stride=1,
                #     block_fn=Block,
                #     name="4prime_stack",
                # )(outputs)
                backbone = tf.keras.Model(inputs=inputs, outputs=outputs)
                # backbone = backbone.as_backbone()
                self.backbone = backbone
        else:
            # TODO(scottzhu): Might need to do more assertion about the model
            if not isinstance(backbone, tf.keras.layers.Layer):
                raise ValueError(
                    "Backbone need to be a `tf.keras.layers.Layer`, "
                    f"received {backbone}"
                )
            self.backbone = backbone

        self._segmentation_head_passed = segmentation_head
        if segmentation_head is None:
            # Scale up the output when using FPN, to keep the output shape same as the
            # input shape.
            output_scale_factor = 16

            segmentation_head = (
                keras_cv.models.segmentation.__internal__.SegmentationHead(
                    classes=classes, convs=0, output_scale_factor=output_scale_factor
                )
            )
        self.segmentation_head = segmentation_head

    def call(self, inputs, training=None):
        backbone_output = self.backbone(inputs, training=training)
        y_pred = self.segmentation_head(backbone_output, training=training)
        return y_pred

    # TODO(tanzhenyu): consolidate how regularization should be applied to KerasCV.
    def compile(self, weight_decay=0.0001, **kwargs):
        self.weight_decay = weight_decay
        super().compile(**kwargs)

    def train_step(self, data):
        images, y_true, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(images, training=True)
            total_loss = self.compute_loss(images, y_true, y_pred, sample_weight)
            reg_losses = []
            if self.weight_decay:
                for var in self.trainable_variables:
                    if "bn" not in var.name:
                        reg_losses.append(self.weight_decay * tf.nn.l2_loss(var))
                l2_loss = tf.math.add_n(reg_losses)
                total_loss += l2_loss
        self.optimizer.minimize(total_loss, self.trainable_variables, tape=tape)
        # tf.print("l2_loss", l2_loss)
        return self.compute_metrics(images, y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        config = {
            "classes": self.classes,
            "backbone": self._backbone_passed,
            "decoder": self._decoder_passed,
            "segmentation_head": self._segmentation_head_passed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
