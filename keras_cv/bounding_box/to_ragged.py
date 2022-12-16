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

from keras_cv.bounding_box import validate


def to_ragged(bounding_boxes):
    """converts a Dense padded bounding box `tf.Tensor` to a `tf.RaggedTensor`.

    Bounding boxes are ragged tensors in most use cases. Converting them to a dense
    tensor makes it easier to work with Tensorflow ecosystem.
    This function can be used to filter out the masked out bounding boxes by
    checking for padded sentinel value of the class_id axis of the bounding_boxes.

    Usage:
    ```python
    bounding_boxes = {
        "boxes": tf.constant([[2, 3, 4, 5], [0, 1, 2, 3]]),
        "classes": tf.constant([[-1, 1]]),
    }
    bounding_boxes = bounding_box.to_ragged(bounding_boxes)
    print(bounding_boxes)
    # {
    #     "boxes": [[0, 1, 2, 3]],
    #     "classes": [[1]]
    # }
    ```

    Args:
        bounding_boxes: a Tensor of bounding boxes.  May be batched, or unbatched.

    Returns:
        dictionary of `tf.RaggedTensor` or 'tf.Tensor' containing the filtered bounding
        boxes.
    """
    info = validate.validate(bounding_boxes)

    if info["ragged"]:
        return bounding_boxes

    boxes = bounding_boxes.get("boxes")
    classes = bounding_boxes.get("classes")
    mask = classes == -1

    if isinstance(boxes, tf.RaggedTensor):
        boxes = boxes.to_tensor(default_value=-1)
    if isinstance(classes, tf.RaggedTensor):
        classes = classes.to_tensor(default_value=-1)

    boxes = tf.ragged.boolean_mask(boxes, mask[None])
    classes = tf.ragged.boolean_mask(classes, mask)
    result = bounding_boxes.copy()
    result["boxes"] = boxes
    result["classes"] = classes
    return result
