from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import six
import tensorflow_datasets as tfds

from pycocotools import coco
from pycocotools import cocoeval
from PIL import Image
import copy
import keras_cv
import keras_cv.layers
from keras_cv import bounding_box
from keras.utils import data_utils


global_batch = 1
image_size = [640, 640, 3]
eval_ds = tfds.load("voc/2007", split="test", with_info=False, shuffle_files=True)


class COCOWrapper(coco.COCO):
  """COCO wrapper class.
  This class wraps COCO API object, which provides the following additional
  functionalities:
    1. Support string type image id.
    2. Support loading the groundtruth dataset using the external annotation
       dictionary.
    3. Support loading the prediction results using the external annotation
       dictionary.
  """

  def __init__(self, gt_dataset=None):
    """Instantiates a COCO-style API object.
    Args:
      eval_type: either 'box' or 'mask'.
      annotation_file: a JSON file that stores annotations of the eval dataset.
        This is required if `gt_dataset` is not provided.
      gt_dataset: the groundtruth eval datatset in COCO API format.
    """

    coco.COCO.__init__(self, annotation_file=None)
    self._eval_type = 'box'
    if gt_dataset:
      self.dataset = gt_dataset
      self.createIndex()

  def loadRes(self, predictions):
    """Loads result file and return a result api object.
    Args:
      predictions: a list of dictionary each representing an annotation in COCO
        format. The required fields are `image_id`, `category_id`, `score`,
        `bbox`, `segmentation`.
    Returns:
      res: result COCO api object.
    Raises:
      ValueError: if the set of image id from predctions is not the subset of
        the set of image id of the groundtruth dataset.
    """
    res = coco.COCO()
    res.dataset['images'] = copy.deepcopy(self.dataset['images'])
    res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

    image_ids = [ann['image_id'] for ann in predictions]
    if set(image_ids) != (set(image_ids) & set(self.getImgIds())):
      raise ValueError('Results do not correspond to the current dataset!')
    for ann in predictions:
      x1, x2, y1, y2 = [ann['bbox'][0], ann['bbox'][0] + ann['bbox'][2],
                        ann['bbox'][1], ann['bbox'][1] + ann['bbox'][3]]

      ann['area'] = ann['bbox'][2] * ann['bbox'][3]
      ann['segmentation'] = [
          [x1, y1, x1, y2, x2, y2, x2, y1]]

    res.dataset['annotations'] = copy.deepcopy(predictions)
    res.createIndex()
    return res

def yxyx_to_xywh(boxes):
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  boxes_ymin = boxes[..., 0]
  boxes_xmin = boxes[..., 1]
  boxes_width = boxes[..., 3] - boxes[..., 1]
  boxes_height = boxes[..., 2] - boxes[..., 0]
  new_boxes = np.stack(
      [boxes_xmin, boxes_ymin, boxes_width, boxes_height], axis=-1)

  return new_boxes

def convert_predictions_to_coco_annotations(predictions):
  coco_predictions = []
  num_batches = len(predictions['source_id'])
  max_num_detections = predictions['detection_classes'][0].shape[1]
  use_outer_box = 'detection_outer_boxes' in predictions
  for i in range(num_batches):
    predictions['detection_boxes'][i] = yxyx_to_xywh(
        predictions['detection_boxes'][i])
    if use_outer_box:
      predictions['detection_outer_boxes'][i] = yxyx_to_xywh(
          predictions['detection_outer_boxes'][i])
      mask_boxes = predictions['detection_outer_boxes']
    else:
      mask_boxes = predictions['detection_boxes']

    batch_size = predictions['source_id'][i].shape[0]
    for j in range(batch_size):
      for k in range(max_num_detections):
        ann = {}
        ann['image_id'] = predictions['source_id'][i][j]
        ann['category_id'] = predictions['detection_classes'][i][j, k]
        ann['bbox'] = predictions['detection_boxes'][i][j, k]
        ann['score'] = predictions['detection_scores'][i][j, k]
        coco_predictions.append(ann)

  for i, ann in enumerate(coco_predictions):
    ann['id'] = i + 1

  return coco_predictions

def convert_groundtruths_to_coco_dataset(groundtruths, label_map=None):
  source_ids = np.concatenate(groundtruths['source_id'], axis=0)
  heights = np.concatenate(groundtruths['height'], axis=0)
  widths = np.concatenate(groundtruths['width'], axis=0)
  gt_images = [{'id': int(i), 'height': int(h), 'width': int(w)} for i, h, w
               in zip(source_ids, heights, widths)]

  gt_annotations = []
  num_batches = len(groundtruths['source_id'])
  for i in range(num_batches):
    max_num_instances = groundtruths['classes'][i].shape[1]
    batch_size = groundtruths['source_id'][i].shape[0]
    for j in range(batch_size):
      num_instances = groundtruths['num_detections'][i][j]
      if num_instances > max_num_instances:
        num_instances = max_num_instances
      for k in range(int(num_instances)):
        ann = {}
        ann['image_id'] = int(groundtruths['source_id'][i][j])
        if 'is_crowds' in groundtruths:
          ann['iscrowd'] = int(groundtruths['is_crowds'][i][j, k])
        else:
          ann['iscrowd'] = 0
        ann['category_id'] = int(groundtruths['classes'][i][j, k])
        boxes = groundtruths['boxes'][i]
        ann['bbox'] = [
            float(boxes[j, k, 1]),
            float(boxes[j, k, 0]),
            float(boxes[j, k, 3] - boxes[j, k, 1]),
            float(boxes[j, k, 2] - boxes[j, k, 0])]
        if 'areas' in groundtruths:
          ann['area'] = float(groundtruths['areas'][i][j, k])
        else:
          ann['area'] = float(
              (boxes[j, k, 3] - boxes[j, k, 1]) *
              (boxes[j, k, 2] - boxes[j, k, 0]))
        gt_annotations.append(ann)

  for i, ann in enumerate(gt_annotations):
    ann['id'] = i + 1

  if label_map:
    gt_categories = [{'id': i, 'name': label_map[i]} for i in label_map]
  else:
    category_ids = [gt['category_id'] for gt in gt_annotations]
    gt_categories = [{'id': i} for i in set(category_ids)]

  gt_dataset = {
      'images': gt_images,
      'categories': gt_categories,
      'annotations': copy.deepcopy(gt_annotations),
  }
  return gt_dataset

class COCOEvaluator(object):
  """COCO evaluation metric class."""

  def __init__(self,
               per_category_metrics=False):
    self._per_category_metrics = per_category_metrics
    self._metric_names = [
        'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1', 'ARmax10',
        'ARmax100', 'ARs', 'ARm', 'ARl'
    ]
    self._required_prediction_fields = [
        'source_id', 'num_detections', 'detection_classes', 'detection_scores',
        'detection_boxes'
    ]
    self._required_groundtruth_fields = [
        'source_id', 'height', 'width', 'classes', 'boxes'
    ]

    self.reset_states()

  def reset_states(self):
    """Resets internal states for a fresh run."""
    self._predictions = {}
    self._groundtruths = {}

  def result(self):
    """Evaluates detection results, and reset_states."""
    metric_dict = self.evaluate()
    # Cleans up the internal variables in order for a fresh eval next time.
    self.reset_states()
    return metric_dict

  def evaluate(self):
    """Evaluates with detections from all images with COCO API.
    Returns:
      coco_metric: float numpy array with shape [24] representing the
        coco-style evaluation metrics (box and mask).
    """
    gt_dataset = convert_groundtruths_to_coco_dataset(
        self._groundtruths)
    coco_gt = COCOWrapper(
        gt_dataset=gt_dataset)
    coco_predictions = convert_predictions_to_coco_annotations(
        self._predictions)
    coco_dt = coco_gt.loadRes(predictions=coco_predictions)
    image_ids = [ann['image_id'] for ann in coco_predictions]

    coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    metrics = coco_metrics

    metrics_dict = {}
    for i, name in enumerate(self._metric_names):
      metrics_dict[name] = metrics[i].astype(np.float32)

    # Adds metrics per category.
    if self._per_category_metrics:
      metrics_dict.update(self._retrieve_per_category_metrics(coco_eval))

    return metrics_dict

  def _retrieve_per_category_metrics(self, coco_eval, prefix=''):
    """Retrieves and per-category metrics and retuns them in a dict.
    Args:
      coco_eval: a cocoeval.COCOeval object containing evaluation data.
      prefix: str, A string used to prefix metric names.
    Returns:
      metrics_dict: A dictionary with per category metrics.
    """

    metrics_dict = {}
    if prefix:
      prefix = prefix + ' '

    if hasattr(coco_eval, 'category_stats'):
      for category_index, category_id in enumerate(coco_eval.params.catIds):
        category_display_name = category_id

        metrics_dict[prefix + 'Precision mAP ByCategory/{}'.format(
            category_display_name
        )] = coco_eval.category_stats[0][category_index].astype(np.float32)
        metrics_dict[prefix + 'Precision mAP ByCategory@50IoU/{}'.format(
            category_display_name
        )] = coco_eval.category_stats[1][category_index].astype(np.float32)
        metrics_dict[prefix + 'Precision mAP ByCategory@75IoU/{}'.format(
            category_display_name
        )] = coco_eval.category_stats[2][category_index].astype(np.float32)
        metrics_dict[prefix + 'Precision mAP ByCategory (small) /{}'.format(
            category_display_name
        )] = coco_eval.category_stats[3][category_index].astype(np.float32)
        metrics_dict[prefix + 'Precision mAP ByCategory (medium) /{}'.format(
            category_display_name
        )] = coco_eval.category_stats[4][category_index].astype(np.float32)
        metrics_dict[prefix + 'Precision mAP ByCategory (large) /{}'.format(
            category_display_name
        )] = coco_eval.category_stats[5][category_index].astype(np.float32)
        metrics_dict[prefix + 'Recall AR@1 ByCategory/{}'.format(
            category_display_name
        )] = coco_eval.category_stats[6][category_index].astype(np.float32)
        metrics_dict[prefix + 'Recall AR@10 ByCategory/{}'.format(
            category_display_name
        )] = coco_eval.category_stats[7][category_index].astype(np.float32)
        metrics_dict[prefix + 'Recall AR@100 ByCategory/{}'.format(
            category_display_name
        )] = coco_eval.category_stats[8][category_index].astype(np.float32)
        metrics_dict[prefix + 'Recall AR (small) ByCategory/{}'.format(
            category_display_name
        )] = coco_eval.category_stats[9][category_index].astype(np.float32)
        metrics_dict[prefix + 'Recall AR (medium) ByCategory/{}'.format(
            category_display_name
        )] = coco_eval.category_stats[10][category_index].astype(np.float32)
        metrics_dict[prefix + 'Recall AR (large) ByCategory/{}'.format(
            category_display_name
        )] = coco_eval.category_stats[11][category_index].astype(np.float32)

    return metrics_dict

  def _convert_to_numpy(self, groundtruths, predictions):
    """Converts tesnors to numpy arrays."""
    if groundtruths:
      labels = tf.nest.map_structure(lambda x: x.numpy(), groundtruths)
      numpy_groundtruths = {}
      for key, val in labels.items():
        if isinstance(val, tuple):
          val = np.concatenate(val)
        numpy_groundtruths[key] = val
    else:
      numpy_groundtruths = groundtruths

    if predictions:
      outputs = tf.nest.map_structure(lambda x: x.numpy(), predictions)
      numpy_predictions = {}
      for key, val in outputs.items():
        if isinstance(val, tuple):
          val = np.concatenate(val)
        numpy_predictions[key] = val
    else:
      numpy_predictions = predictions

    return numpy_groundtruths, numpy_predictions

  def update_state(self, groundtruths, predictions):
    """Update and aggregate detection results and groundtruth data.
    Args:
      groundtruths: a dictionary of Tensors including the fields below.
        See also different parsers under `../dataloader` for more details.
        Required fields:
          - source_id: a numpy array of int or string of shape [batch_size].
          - height: a numpy array of int of shape [batch_size].
          - width: a numpy array of int of shape [batch_size].
          - num_detections: a numpy array of int of shape [batch_size].
          - boxes: a numpy array of float of shape [batch_size, K, 4].
          - classes: a numpy array of int of shape [batch_size, K].
        Optional fields:
          - is_crowds: a numpy array of int of shape [batch_size, K]. If the
              field is absent, it is assumed that this instance is not crowd.
          - areas: a numy array of float of shape [batch_size, K]. If the
              field is absent, the area is calculated using either boxes or
              masks depending on which one is available.
          - masks: a numpy array of float of shape
              [batch_size, K, mask_height, mask_width],
      predictions: a dictionary of tensors including the fields below.
        See different parsers under `../dataloader` for more details.
        Required fields:
          - source_id: a numpy array of int or string of shape [batch_size].
          - image_info [if `need_rescale_bboxes` is True]: a numpy array of
            float of shape [batch_size, 4, 2].
          - num_detections: a numpy array of
            int of shape [batch_size].
          - detection_boxes: a numpy array of float of shape [batch_size, K, 4].
          - detection_classes: a numpy array of int of shape [batch_size, K].
          - detection_scores: a numpy array of float of shape [batch_size, K].
        Optional fields:
          - detection_masks: a numpy array of float of shape
              [batch_size, K, mask_height, mask_width].
    Raises:
      ValueError: if the required prediction or groundtruth fields are not
        present in the incoming `predictions` or `groundtruths`.
    """
    groundtruths, predictions = self._convert_to_numpy(groundtruths,
                                                       predictions)
    for k in self._required_prediction_fields:
      if k not in predictions:
        raise ValueError(
            'Missing the required key `{}` in predictions!'.format(k))
    for k, v in six.iteritems(predictions):
      if k not in self._predictions:
        self._predictions[k] = [v]
      else:
        self._predictions[k].append(v)

    assert groundtruths
    for k in self._required_groundtruth_fields:
      if k not in groundtruths:
        raise ValueError(
            'Missing the required key `{}` in groundtruths!'.format(k))
    for k, v in six.iteritems(groundtruths):
      if k not in self._groundtruths:
        self._groundtruths[k] = [v]
      else:
        self._groundtruths[k].append(v)


def proc_eval_fn(bounding_box_format, img_size):
    resizing = tf.keras.layers.Resizing(
        height=img_size[0], width=img_size[1], crop_to_aspect_ratio=False
    )

    def apply(inputs):
        source_id = tf.strings.to_number(
            tf.strings.split(inputs["image/filename"], '.')[0], tf.int64)
        raw_image = inputs["image"]
        raw_image = tf.cast(raw_image, tf.float32)
        image = tf.keras.applications.resnet50.preprocess_input(raw_image)
        image = resizing(image)
        raw_image = resizing(raw_image)
        gt_boxes = keras_cv.bounding_box.convert_format(
            inputs["objects"]["bbox"],
            images=image,
            source="rel_yxyx",
            target=bounding_box_format,
        )
        gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
        gt_classes = tf.expand_dims(gt_classes, axis=-1)
        return {
            "source_id": source_id,
            "raw_images": raw_image,
            "images": image,
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


eval_ds = eval_ds.map(
    proc_eval_fn("yxyx", [640, 640, 3]), num_parallel_calls=tf.data.AUTOTUNE
)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)
eval_ds = eval_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
model = keras_cv.models.FasterRCNN(classes=20, bounding_box_format="yxyx")

examples = next(iter(eval_ds))
outputs = model(
    images=examples["images"],
    gt_boxes=examples["gt_boxes"],
    gt_classes=examples["gt_classes"],
    training=True,
)
weights = data_utils.get_file(origin="https://storage.googleapis.com/keras-cv/models/fasterrcnn/voc/weights_final.h5", cache_subdir="models")
model.load_weights(weights)

mAP = keras_cv.metrics.COCOMeanAveragePrecision(
    class_ids=range(20),
    bounding_box_format="yxyx",
    # not sure what recall means in precision compute
    recall_thresholds=[x / 100.0 for x in range(50, 100, 5)],
    name="Mean Average Precision",
)

def visualize_image(images, pred_boxes, gt_boxes):
    gt_color = tf.constant(((255.0, 0, 0),))
    pred_color = tf.constant(((0.0, 255.0, 0.0),))
    plt.figure(figsize=(10, 10))
    pred_boxes = bounding_box.convert_format(
        boxes=pred_boxes, source="yxyx", target="rel_yxyx", image_shape=image_size
    )
    gt_boxes = bounding_box.convert_format(
        boxes=gt_boxes, source="yxyx", target="rel_yxyx", image_shape=image_size
    )
    plotted_images = tf.image.draw_bounding_boxes(images, pred_boxes, pred_color)
    plt.subplot(2, 1, 1)
    plt.imshow(plotted_images[0].numpy().astype("uint8"))
    plt.subplot(2, 1, 2)
    plotted_images = tf.image.draw_bounding_boxes(images, gt_boxes, gt_color)
    plt.imshow(plotted_images[0].numpy().astype("uint8"))
    plt.show()

evaluator = COCOEvaluator()

def eval_map_step(examples):
    images = examples["images"]
    outputs = model(images=images, training=False)
    gt_boxes = examples["gt_boxes"]
    gt_classes = examples["gt_classes"]
    box_pred = outputs["rcnn_box_pred"]
    cls_pred = outputs["rcnn_cls_pred"]
    box_pred = tf.expand_dims(box_pred, axis=-2)
    box_pred, scores_pred, cls_pred, valid_det = tf.image.combined_non_max_suppression(
        boxes=box_pred,
        scores=cls_pred[..., :-1],
        max_output_size_per_class=10,
        max_total_size=10,
        score_threshold=0.5,
        iou_threshold=0.5,
        clip_boxes=False,
    )
    valid = valid_det.numpy()[0]

    box_pred = box_pred[:, :valid, :]
    scores_pred = scores_pred[:, :valid, tf.newaxis]
    cls_pred = cls_pred[:, :valid, tf.newaxis]
    y_true = tf.concat([gt_boxes, gt_classes], axis=-1)
    y_pred = tf.concat([box_pred, cls_pred, scores_pred], axis=-1)
    mAP.update_state(y_true, y_pred)

    # visualize_image(examples["raw_images"], box_pred, gt_boxes)


def eval_pycoco_step(examples):
    images = examples["images"]
    outputs = model(images=images, training=False)
    gt_boxes = examples["gt_boxes"]
    gt_classes = examples["gt_classes"]
    box_pred = outputs["rcnn_box_pred"]
    cls_pred = outputs["rcnn_cls_pred"]
    box_pred = tf.expand_dims(box_pred, axis=-2)
    box_pred, scores_pred, cls_pred, valid_det = tf.image.combined_non_max_suppression(
        boxes=box_pred,
        scores=cls_pred[..., :-1],
        max_output_size_per_class=10,
        max_total_size=10,
        score_threshold=0.5,
        iou_threshold=0.5,
        clip_boxes=False,
    )

    visualize_image(examples["raw_images"], box_pred, gt_boxes)

    ground_truth = {}
    ground_truth["source_id"] = examples["source_id"]
    ground_truth["height"] = tf.tile(tf.constant([640]), [global_batch])
    ground_truth["width"] = tf.tile(tf.constant([640]), [global_batch])
    num_dets = gt_classes.get_shape().as_list()[1]
    ground_truth["num_detections"] = tf.tile(tf.constant([num_dets]), [global_batch])
    ground_truth["boxes"] = gt_boxes
    ground_truth["classes"] = gt_classes

    predictions = {}
    predictions["source_id"] = examples["source_id"]
    predictions["num_detections"] = valid_det
    predictions["detection_boxes"] = box_pred
    predictions["detection_classes"] = cls_pred
    predictions["detection_scores"] = scores_pred

    evaluator.update_state(ground_truth, predictions)

print("start {}".format(datetime.now()))
for examples in eval_ds:
    eval_pycoco_step(examples)
    # eval_map_step(examples)

# Using COCOWrapper takes 8 minutes with mAP=0.443
print("MAP {}".format(evaluator.result()))

# Using keras_cv COCOMap takes 13 minutes with mAP=0.187
# print("MAP {}".format(mAP.result()))
print("end {}".format(datetime.now()))