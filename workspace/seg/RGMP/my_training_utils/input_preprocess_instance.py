#Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Prepares the data used for DeepLab training/evaluation."""
import tensorflow as tf
from deeplab.core import feature_extractor
from deeplab.core import preprocess_utils


# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5


def preprocess_image_and_label(image,
                               label,
                               target_height,
                               target_width,
                               inner_extension_ratio=-0.1,
                               outer_extension_ratio=0.2,
                               filling='central_padding',
                               ignore_label=255,
                               is_training=True,
                               model_variant=None):
  """Preprocesses the image and label for instance seg.
  Args:
    image
    label
    target_height
    target_width
    inner_extension_ratio in (-0.5, 0.5)
    outer_extension_ratio in (-0.5, 0.5) and >= inner_extension_ratio
    filling = 'central_padding' or 'scaling'
    is_training
    ignore_label
    model_variant
  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.
  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')
  if outer_extension_ratio < inner_extension_ratio:
    raise ValueError('outer_extension_ratio >= inner_extension_ratio must hold.')

  # Keep reference to original image.
  original_image = image

  processed_image = tf.cast(image, tf.float32)
  if label is not None:
    label = tf.cast(label, tf.int32)

  # 0. get true bbox and restore position pixels
  image_shape = tf.shape(processed_image)
  image_height = image_shape[0]
  image_width = image_shape[1]

  pos_pixels_pos = [[[0,0], [0,1], [0,2]],
                    [[image_height - 1, 0], [image_height - 1, 1], [image_height - 1, 2]],
                    [[0, image_width - 3], [0, image_width - 2], [0, image_width - 1]],
                    [[image_height - 1, image_width - 3], 
                     [image_height - 1, image_width - 2],
                     [image_height - 1, image_width - 1]]]

  # calculate bbox
#  bbox = [86, 788, 110, 329]  # hmin, hmax, wmin, wmax
  bbox = [0, 0, 0, 0]
  count = 0
  for triplet in pos_pixels_pos:
    # get bbox position
    high6 = label[triplet[0][0], triplet[0][1], 0] / 4
    mid6 = label[triplet[1][0], triplet[1][1], 0] / 4
    low6 = label[triplet[2][0], triplet[2][1], 0] / 4
    bbox[count] = high6 * 64 * 64 + mid6 * 64 + low6
    count += 1

  # restore label
  mask = tf.concat(
      [[tf.concat([[1,1,1], tf.zeros([image_width-6], dtype=tf.int32), [1,1,1]], 0)],
       tf.zeros([image_height-2, image_width], dtype=tf.int32), 
       [tf.concat([[1,1,1], tf.zeros([image_width-6], dtype=tf.int32), [1,1,1]], 0)]], 
      0)
  mask = tf.expand_dims(mask, 2)

  ignore_label_tensor = tf.zeros_like(label, dtype=tf.int32) + ignore_label
  original_label = tf.where(tf.floormod(label, 4) > 1,
                            ignore_label_tensor, 
                            tf.floormod(label, 4))
 
  label = tf.multiply(label, 1 - mask) + tf.multiply(original_label, mask)

  # 1. limited random crop
  bbox_height = tf.to_float(bbox[1] - bbox[0])
  bbox_width = tf.to_float(bbox[3] - bbox[2])
  inner_extension_ratio = tf.to_float(inner_extension_ratio)
  outer_extension_ratio = tf.to_float(outer_extension_ratio)
#  crop_bbox = tf.convert_to_tensor(bbox, dtype=tf.int32)
  crop_bbox = [0, 0, 0, 0]  # hmin, hmax, wmin, wmax
  crop_bbox[0] = tf.random_uniform(
     [],  # 16, 87
     tf.maximum(0, bbox[0] - tf.to_int32(bbox_height * outer_extension_ratio)), 
     tf.maximum(0, bbox[0] - tf.to_int32(bbox_height * inner_extension_ratio)) + 1, 
     dtype=tf.int32)
  crop_bbox[1] = tf.random_uniform(
     [], # 788, 859
     tf.minimum(image_shape[0], bbox[1] + tf.to_int32(bbox_height * inner_extension_ratio)),
     tf.minimum(image_shape[0], bbox[1] + tf.to_int32(bbox_height * outer_extension_ratio)) + 1, 
     dtype=tf.int32)
  crop_bbox[2] = tf.random_uniform(
     [],  # 89, 111
     tf.maximum(0, bbox[2] - tf.to_int32(bbox_width * outer_extension_ratio)),
     tf.maximum(0, bbox[2] - tf.to_int32(bbox_width * inner_extension_ratio)) + 1, 
     dtype=tf.int32)
  crop_bbox[3] = tf.random_uniform(
     [],  # 329, 351
     tf.minimum(image_shape[1], bbox[3] + tf.to_int32(bbox_width * inner_extension_ratio)),
     tf.minimum(image_shape[1], bbox[3] + tf.to_int32(bbox_width * outer_extension_ratio)) + 1, 
     dtype=tf.int32)
  crop_bbox = tf.convert_to_tensor(crop_bbox, dtype=tf.int32)

  processed_image = processed_image[crop_bbox[0]: crop_bbox[1], 
                                    crop_bbox[2]: crop_bbox[3], :]
  label = label[crop_bbox[0]: crop_bbox[1],
                crop_bbox[2]: crop_bbox[3], :]
  cropped_height = crop_bbox[1] - crop_bbox[0]
  cropped_width = crop_bbox[3] - crop_bbox[2]

  # 2. central padding or resize
  if filling == 'central_padding':
    [processed_image, label] = (
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=tf.minimum(target_height, target_width),
            max_size=tf.maximum(target_height, target_width),
            factor=None,
            align_corners=True))
    scaled_shape = tf.shape(processed_image)
    scaled_height = scaled_shape[0]
    scaled_width = scaled_shape[1]

    mean_pixel = tf.reshape(
        feature_extractor.mean_pixel(model_variant), [1, 1, 3])
    processed_image = preprocess_utils.pad_to_bounding_box(
        processed_image, 
        (target_height - scaled_height) / 2, (target_width - scaled_width) / 2, 
        target_height, target_width,
        mean_pixel)
    if label is not None:
      label = preprocess_utils.pad_to_bounding_box(
          label,
          (target_height - scaled_height) / 2, (target_width - scaled_width) / 2,
          target_height, target_width,
          ignore_label)
  elif filling == 'resize':
    processed_image = tf.image.resize_images(
        processed_image, [target_height, target_width], method=tf.image.ResizeMethod.BILINEAR)
    if label is not None:
      label = tf.image.resize_images(
          label, [target_height, target_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  else:
    raise ValueError('Filling method not recognized.')

  processed_image.set_shape([target_height, target_width, 3])
  if label is not None:
    label.set_shape([target_height, target_width, 1])
 
  # 3. random flip
  if is_training:
    # Randomly left-right flip the image and label.
    processed_image, label, _ = preprocess_utils.flip_dim(
        [processed_image, label], _PROB_OF_FLIP, dim=1)

  return original_image, processed_image, label  # , crop_bbox
