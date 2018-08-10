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
"""Wrapper for providing semantic segmentation data."""

import tensorflow as tf
from deeplab import common
from my_data_utils import input_preprocess_video
from my_training_utils import input_preprocess_instance

slim = tf.contrib.slim

dataset_data_provider = slim.dataset_data_provider



def _get_data(data_provider, dataset_split):
  """Gets data from data provider.
  Args:
    data_provider: An object of slim.data_provider.
    dataset_split: Dataset split.
  Returns:
    image: Image Tensor.
    label: Label Tensor storing segmentation annotations.
    image_name: Image name.
    height: Image height.
    width: Image width.
  Raises:
    ValueError: Failed to find label.
  """
  if common.LABELS_CLASS0 not in data_provider.list_items():
    raise ValueError('Failed to find labels.')

  image0, image1, image2, image3, image4, image5, image6, image7, image8, image9, height, width = data_provider.get(
      [common.IMAGE0, common.IMAGE1, common.IMAGE2, common.IMAGE3, common.IMAGE4, common.IMAGE5, common.IMAGE6, common.IMAGE7, common.IMAGE8, common.IMAGE9, common.HEIGHT, common.WIDTH])
  last_mask, first_image, first_mask = data_provider.get(
      [common.LAST_MASK, common.FIRST_IMAGE, common.FIRST_MASK])
  # Some datasets do not contain image_name.
  if common.IMAGE_NAME in data_provider.list_items():
    image_name, = data_provider.get([common.IMAGE_NAME])
  else:
    image_name = tf.constant('')

  label = None
  if dataset_split != common.TEST_SET:
    label0, label1, label2, label3, label4, label5, label6, label7, label8, label9 = data_provider.get(
      [common.LABELS_CLASS0, common.LABELS_CLASS1, common.LABELS_CLASS2, common.LABELS_CLASS3, common.LABELS_CLASS4, common.LABELS_CLASS5, common.LABELS_CLASS6, common.LABELS_CLASS7, common.LABELS_CLASS8, common.LABELS_CLASS9])

  image = tf.concat([first_image,first_mask,image0,last_mask,image1,image2,image3,image4,image5,image6,image7,image8,image9],2)
  label = tf.concat([label0, label1, label2, label3, label4, label5, label6, label7, label8, label9],2)

  print('image shape:')
  print(image.shape)
  print('label shape:')
  print(label.shape)
  
  return image, label, image_name, height, width


def get(dataset,
        crop_size,
        batch_size,
        min_resize_value=None,
        max_resize_value=None,
        resize_factor=None,
        min_scale_factor=1.,
        max_scale_factor=1.,
        scale_factor_step_size=0,
        num_readers=1,
        num_threads=1,
        dataset_split=None,
        is_training=True,
        model_variant=None,
        instance_seg=False,
        instance_seg_args={},
        test=False):
  """Gets the dataset split for semantic segmentation.
  This functions gets the dataset split for semantic segmentation. In
  particular, it is a wrapper of (1) dataset_data_provider which returns the raw
  dataset split, (2) input_preprcess which preprocess the raw data, and (3) the
  Tensorflow operation of batching the preprocessed data. Then, the output could
  be directly used by training, evaluation or visualization.
  Args:
    dataset: An instance of slim Dataset.
    crop_size: Image crop size [height, width].
    batch_size: Batch size.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    num_readers: Number of readers for data provider.
    num_threads: Number of threads for batching data.
    dataset_split: Dataset split.
    is_training: Is training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.
  Returns:
    A dictionary of batched Tensors for semantic segmentation.
  Raises:
    ValueError: dataset_split is None, failed to find labels, or label shape
      is not valid.
  """
  if dataset_split is None:
    raise ValueError('Unknown dataset split.')
  if model_variant is None:
    tf.logging.warning('Please specify a model_variant. See '
                       'feature_extractor.network_map for supported model '
                       'variants.')

  data_provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      num_epochs=None if is_training else 1,
      shuffle=is_training)
  image, label, image_name, height, width = _get_data(data_provider,
                                                      dataset_split)
  if label is not None:
    if label.shape.ndims == 3 and label.shape.dims[2] == 10:
      pass
    else:
      raise ValueError('Input label shape must be [height, width], or '
                       '[height, width, 10].')

    label.set_shape([None, None, 10])


  # if test==True:
  #   sample = {
  #     common.IMAGE: image,
  #     common.IMAGE_NAME: image_name,
  #     common.HEIGHT: height,
  #     common.WIDTH: width
  #    }
  #   return sample

  if instance_seg:
    inner_extension_ratio = instance_seg_args['inner_extension_ratio']
    outer_extension_ratio = instance_seg_args['outer_extension_ratio']
    filling = instance_seg_args['filling']
    original_image, image, label = input_preprocess_instance.preprocess_image_and_label(
        image,
        label,
        crop_size[0],
        crop_size[1],
        inner_extension_ratio=inner_extension_ratio,
        outer_extension_ratio=outer_extension_ratio,
        filling=filling,
        ignore_label=dataset.ignore_label,
        is_training=is_training,
        model_variant=model_variant)
  else:
    original_image, image, label = input_preprocess_video.preprocess_image_and_label(
        image,
        label,
        crop_height=crop_size[0],
        crop_width=crop_size[1],
        min_resize_value=min_resize_value,
        max_resize_value=max_resize_value,
        resize_factor=resize_factor,
        min_scale_factor=min_scale_factor,
        max_scale_factor=max_scale_factor,
        scale_factor_step_size=scale_factor_step_size,
        ignore_label=dataset.ignore_label,
        is_training=is_training,
        model_variant=model_variant)


  sample = {
      common.IMAGE: image,
      common.IMAGE_NAME: image_name,
      common.HEIGHT: height,
      common.WIDTH: width
  }
  if label is not None:
    sample[common.LABEL] = label

  if not is_training:
    # Original image is only used during visualization.
    sample[common.ORIGINAL_IMAGE] = original_image,
    num_threads = 1

  

  return tf.train.batch(
      sample,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=32 * batch_size,
      allow_smaller_final_batch=not is_training,
      dynamic_pad=True)
