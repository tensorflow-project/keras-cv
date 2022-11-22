import sys

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import tensorflow_datasets as tfds

import keras_cv
import keras_cv.layers
from keras_cv import bounding_box
from keras_cv.bounding_box.converters import convert_format
from keras_cv.bounding_box.iou import compute_iou
from keras_cv.layers.object_detection import anchor_generator

numpy.set_printoptions(threshold=sys.maxsize)

global_batch = 4

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices[1:], "GPU")

eval_ds = tfds.load("voc/2012", split="validation", with_info=False, shuffle_files=True)

image_size = [640, 640, 3]
model = keras_cv.models.FasterRCNN(classes=20, bounding_box_format="xyxy")


def resize_and_crop_image(
    image,
    desired_size,
    padded_size,
    aug_scale_min=1.0,
    aug_scale_max=1.0,
    seed=1,
    method=tf.image.ResizeMethod.BILINEAR,
):
    with tf.name_scope("resize_and_crop_image"):
        image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

        random_jittering = aug_scale_min != 1.0 or aug_scale_max != 1.0

        if random_jittering:
            random_scale = tf.random.uniform(
                [], aug_scale_min, aug_scale_max, seed=seed
            )
            scaled_size = tf.round(random_scale * desired_size)
        else:
            scaled_size = desired_size

        scale = tf.minimum(
            scaled_size[0] / image_size[0], scaled_size[1] / image_size[1]
        )
        scaled_size = tf.round(image_size * scale)

        # Computes 2D image_scale.
        image_scale = scaled_size / image_size

        # Selects non-zero random offset (x, y) if scaled image is larger than
        # desired_size.
        if random_jittering:
            max_offset = scaled_size - desired_size
            max_offset = tf.where(
                tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset
            )
            offset = max_offset * tf.random.uniform(
                [
                    2,
                ],
                0,
                1,
                seed=seed,
            )
            offset = tf.cast(offset, tf.int32)
        else:
            offset = tf.zeros((2,), tf.int32)

        scaled_image = tf.image.resize(
            image, tf.cast(scaled_size, tf.int32), method=method
        )

        if random_jittering:
            scaled_image = scaled_image[
                offset[0] : offset[0] + desired_size[0],
                offset[1] : offset[1] + desired_size[1],
                :,
            ]

        output_image = tf.image.pad_to_bounding_box(
            scaled_image, 0, 0, padded_size[0], padded_size[1]
        )

        image_info = tf.stack(
            [
                image_size,
                tf.constant(desired_size, dtype=tf.float32),
                image_scale,
                tf.cast(offset, tf.float32),
            ]
        )
        return output_image, image_info


def resize_and_crop_boxes(boxes, image_scale, output_size, offset):
    with tf.name_scope("resize_and_crop_boxes"):
        # Adjusts box coordinates based on image_scale and offset.
        boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
        boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
        # Clips the boxes.
        boxes = clip_boxes(boxes, output_size)
        return boxes


def clip_boxes(boxes, image_shape):
    if boxes.shape[-1] != 4:
        raise ValueError(
            "boxes.shape[-1] is {:d}, but must be 4.".format(boxes.shape[-1])
        )

    with tf.name_scope("clip_boxes"):
        if isinstance(image_shape, list) or isinstance(image_shape, tuple):
            height, width = image_shape
            max_length = [height, width, height, width]
        else:
            image_shape = tf.cast(image_shape, dtype=boxes.dtype)
            height, width = tf.unstack(image_shape, axis=-1)
            max_length = tf.stack([height, width, height, width], axis=-1)

        clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
        return clipped_boxes


def get_non_empty_box_indices(boxes):
    # Selects indices if box height or width is 0.
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    indices = tf.where(tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
    return indices[:, 0]


def resize_fn(image, gt_boxes, gt_classes):
    image, image_info = resize_and_crop_image(
        image, image_size[:2], image_size[:2], 0.8, 1.25
    )
    gt_boxes = resize_and_crop_boxes(
        gt_boxes, image_info[2, :], image_info[1, :], image_info[3, :]
    )
    indices = get_non_empty_box_indices(gt_boxes)
    gt_boxes = tf.gather(gt_boxes, indices)
    gt_classes = tf.gather(gt_classes, indices)
    return image, gt_boxes, gt_classes


def flip_fn(image, boxes):
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32) > 0.5:
        image = tf.image.flip_left_right(image)
        y1, x1, y2, x2 = tf.split(boxes, num_or_size_splits=4, axis=-1)
        boxes = tf.concat([y1, 1.0 - x2, y2, 1.0 - x1], axis=-1)
    return image, boxes


def proc_train_fn(bounding_box_format, img_size):
    anchors = model.anchor_generator(image_shape=img_size)
    anchors = tf.concat(tf.nest.flatten(anchors), axis=0)
    resizing = tf.keras.layers.Resizing(
        height=img_size[0], width=img_size[1], crop_to_aspect_ratio=False
    )

    def apply(inputs):
        image = inputs["image"]
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        # image = resizing(image)
        raw_image = inputs["image"]
        raw_image = resizing(raw_image)
        gt_boxes = inputs["objects"]["bbox"]
        # image, gt_boxes = flip_fn(image, gt_boxes)
        gt_boxes = keras_cv.bounding_box.convert_format(
            gt_boxes,
            images=image,
            source="rel_yxyx",
            target=bounding_box_format,
        )
        gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
        image, gt_boxes, gt_classes = resize_fn(image, gt_boxes, gt_classes)
        gt_classes = tf.expand_dims(gt_classes, axis=-1)
        box_targets, box_weights, cls_targets, cls_weights = model.rpn_labeler(
            anchors, gt_boxes, gt_classes
        )
        return {
            "raw_images": raw_image,
            "images": image,
            "rpn_box_targets": box_targets,
            "rpn_box_weights": box_weights,
            "rpn_cls_targets": cls_targets,
            "rpn_cls_weights": cls_weights,
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes,
        }

    return apply


def pad_fn(examples):
    gt_boxes = examples.pop("gt_boxes")
    gt_classes = examples.pop("gt_classes")
    examples["gt_boxes"] = gt_boxes.to_tensor(default_value=-1.0)
    examples["gt_classes"] = gt_classes.to_tensor(default_value=-1.0)
    return examples


def filter_fn(examples):
    gt_boxes = examples["objects"]["bbox"]
    if tf.shape(gt_boxes)[0] <= 0 or tf.reduce_sum(gt_boxes) < 0:
        return False
    else:
        return True


def visualize_image(images, pred_boxes=None, gt_boxes=None):
    gt_color = tf.constant(((255.0, 0, 0),))
    pred_color = tf.constant(((0.0, 255.0, 0.0),))
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    if pred_boxes is not None:
        pred_boxes = bounding_box.convert_format(
            boxes=pred_boxes, source="yxyx", target="rel_yxyx", image_shape=image_size
        )
        plotted_images = tf.image.draw_bounding_boxes(images, pred_boxes, pred_color)
        plt.imshow(plotted_images[0].numpy().astype("uint8"))
    plt.subplot(2, 1, 2)
    if gt_boxes is not None:
        gt_boxes = bounding_box.convert_format(
            boxes=gt_boxes, source="yxyx", target="rel_yxyx", image_shape=image_size
        )
        plotted_images = tf.image.draw_bounding_boxes(images, gt_boxes, gt_color)
        plt.imshow(plotted_images[0].numpy().astype("uint8"))
    plt.show()


eval_ds = eval_ds.filter(filter_fn)
eval_ds = eval_ds.map(
    proc_train_fn(bounding_box_format="yxyx", img_size=image_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)

examples = next(iter(eval_ds))
outputs = model(
    images=examples["images"][tf.newaxis, ...],
    gt_boxes=examples["gt_boxes"][tf.newaxis, ...],
    gt_classes=examples["gt_classes"][tf.newaxis, ...],
    training=True,
)
model.load_weights("./weights_1.h5")
print("model output names {}".format(model.output_names))

anchors = model.anchor_generator(image_shape=image_size)
anchors = tf.concat(tf.nest.flatten(anchors), axis=0)

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
cls_loss = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
)

step = 0

# map_fn = proc_train_fn("yxyx", image_size)

# for examples in eval_ds.take(1):
#     print(f'----------------step{step}---------------')
#     examples = map_fn(examples)
#     gt_boxes = examples["gt_boxes"]
#     rpn_box_targets = examples["rpn_box_targets"]
#     rpn_box_weights = examples["rpn_box_weights"][..., 0]
#     num_pos = tf.cast(tf.reduce_sum(rpn_box_weights), tf.int32)
#     print("num positive boxes {}".format(num_pos))
#     _, indices = tf.nn.top_k(rpn_box_weights, k=num_pos)
#     print("anchor indices {}".format(indices))
#     box_targets = model.decode(rpn_box_targets, indices)
#     box_targets = bounding_box.convert_format(box_targets, "center_yxhw", "yxyx")
#     box_targets = tf.gather(box_targets, indices)
#     print("gt boxes {}".format(gt_boxes))
#     print("decoded box_targets {}".format(box_targets))
#     step += 1


for examples in eval_ds.take(10):
    print(f"----------------step{step}---------------")
    outputs = model(
        tf.expand_dims(examples["images"], 0),
        tf.expand_dims(examples["gt_boxes"], 0),
        tf.expand_dims(examples["gt_classes"], 0),
        training=True,
    )
    rpn_box_targets = examples["rpn_box_targets"]
    rpn_box_weights = examples["rpn_box_weights"][..., 0]
    rpn_cls_weights = examples["rpn_cls_weights"][..., 0]

    decoded_box_targets = model.decode(rpn_box_targets, image_shape=image_size)
    decoded_box_targets = convert_format(decoded_box_targets, "center_yxhw", "yxyx")
    rpn_box_pred = outputs["rpn_box_pred"][0, ...]
    rpn_cls_pred = outputs["rpn_cls_pred"][0, ...]
    decoded_box_pred = model.decode(rpn_box_pred, image_shape=image_size)
    decoded_box_pred = convert_format(decoded_box_pred, "center_yxhw", "yxyx")
    num_pos = tf.cast(tf.reduce_sum(rpn_box_weights), tf.int32)
    _, indices = tf.nn.top_k(rpn_box_weights, k=num_pos)
    box_targets = tf.gather(rpn_box_targets, indices)
    box_pred = tf.gather(rpn_box_pred, indices)
    decoded_box_targets = tf.gather(decoded_box_targets, indices)
    decoded_box_pred = tf.gather(decoded_box_pred, indices)
    box_weights = tf.gather(rpn_box_weights, indices)
    pos_cls_pred = tf.gather(rpn_cls_pred, indices)
    num_samples = model.rpn_labeler.samples_per_image
    _, sample_indices = tf.nn.top_k(rpn_cls_weights, k=num_samples)
    cls_pred = tf.gather(rpn_cls_pred, sample_indices)
    print("box weights {}".format(tf.reduce_sum(box_weights)))
    print("num pos {}".format(num_pos))
    print("positive indices {}".format(indices))
    print("gt boxes {}".format(examples["gt_boxes"]))
    print("pos cls pred {}".format(pos_cls_pred))
    # print("cls pred {}".format(cls_pred))
    print("decoded box targets {}".format(decoded_box_targets))
    print("decoded box pred {}".format(decoded_box_pred))
    print("box targets {}".format(tf.gather(rpn_box_targets, indices)))
    print("box pred {}".format(tf.gather(rpn_box_pred, indices)))
    loss = huber_loss(box_targets, box_pred)
    print("loss {}".format(loss))
    loss = tf.reduce_sum(huber_loss(box_targets, box_pred, box_weights))
    # positive_boxes = tf.reduce_sum(box_weights) + 0.01
    loss /= model.rpn_labeler.samples_per_image * global_batch * 0.25
    print("weighted loss {}".format(loss))
    rpn_scores = outputs["rpn_scores"]
    num_negs = tf.reduce_sum(tf.cast(rpn_scores < 0, tf.int32))
    print(
        "pos rpn scores {}".format(tf.reduce_sum(tf.cast(rpn_scores > 0, tf.float32)))
    )
    print("neg rpn scores {}".format(num_negs))
    negative_scores = tf.cast(rpn_scores < 0, tf.float32)[..., 0]
    _, neg_indices = tf.nn.top_k(negative_scores, k=num_negs)
    # print("neg indices {}".format(neg_indices))
    step += 1
