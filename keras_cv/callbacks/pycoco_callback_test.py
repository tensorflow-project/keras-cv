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

import pytest
import tensorflow as tf

import keras_cv
from keras_cv.callbacks import PyCOCOCallback
from keras_cv.metrics.coco.pycoco_wrapper import METRIC_NAMES
from keras_cv.models.object_detection.__test_utils__ import _create_bounding_box_dataset


class PyCOCOCallbackTest(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        yield
        tf.keras.backend.clear_session()

    def test_model_fit_retinanet(self):
        model = keras_cv.models.RetinaNet(
            classes=10,
            bounding_box_format="xywh",
            backbone="resnet50",
            backbone_weights=None,
            include_rescaling=True,
        )
        # all metric formats must match
        model.compile(
            optimizer="adam",
            box_loss="smoothl1",
            classification_loss="focal",
        )

        train_ds = _create_bounding_box_dataset(
            bounding_box_format="xyxy", use_dictionary_box_format=True
        )
        val_ds = _create_bounding_box_dataset(
            bounding_box_format="xyxy", use_dictionary_box_format=True
        )

        def convert_dict_fn(x, y):
            converted_y = {}
            converted_y["boxes"] = y.pop("gt_boxes")
            converted_y["classes"] = y.pop("gt_classes")
            return x, converted_y

        train_ds = train_ds.map(convert_dict_fn)
        val_ds = val_ds.map(convert_dict_fn)

        callback = PyCOCOCallback(validation_data=val_ds, bounding_box_format="xyxy")
        history = model.fit(train_ds, callbacks=[callback])

        self.assertAllInSet(
            [f"val_{metric}" for metric in METRIC_NAMES], history.history.keys()
        )

    def test_input_nms_false(self):
        model = keras_cv.models.FasterRCNN(
            classes=10,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet50V2(
                include_top=False, include_rescaling=True, weights=None
            ).as_backbone(),
        )
        model.compile(
            optimizer="adam",
            box_loss="Huber",
            classification_loss="SparseCategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
        )
        train_ds = _create_bounding_box_dataset(
            bounding_box_format="yxyx", use_dictionary_box_format=True
        )
        eval_ds = _create_bounding_box_dataset(
            bounding_box_format="yxyx", use_dictionary_box_format=True
        )
        callback = PyCOCOCallback(
            validation_data=eval_ds,
            bounding_box_format="yxyx",
            input_nms=False,
        )
        history = model.fit(train_ds, callbacks=[callback])
        self.assertAllInSet(
            [f"val_{metric}" for metric in METRIC_NAMES], history.history.keys()
        )
