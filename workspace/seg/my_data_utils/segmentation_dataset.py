"""
Provides data from semantic segmentation datasets.
Adjusted from models/research/deeplab/datasets/segmentation_dataset.py
"""
import collections
import os.path
import tensorflow as tf

slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['splits_to_sizes',   # Splits of the dataset into training, val, and test.
     'num_classes',   # Number of semantic classes.
     'ignore_label',  # Ignore label value.
    ]
)

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 2975,
        'val': 500,
    },
    num_classes=19,
    ignore_label=255,
)

_PASCAL_VOC_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1464,
        'trainval': 2913,
        'val': 1449,
    },
    num_classes=21,
    ignore_label=255,
)

# These number (i.e., 'train'/'test') seems to have to be hard coded
# You are required to figure it out for your training/testing example.
_ADE20K_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'train': 20210, # num of samples in images/training
        'val': 2000, # num of samples in images/validation
    },
    num_classes=150,
    ignore_label=255,
)

_OWLII_STUDIO_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'train': 6841,
        'val': 2942,
    },
    num_classes=2,
    ignore_label=255,
)

_PASCAL_VOC_HUMAN = DatasetDescriptor(
    splits_to_sizes = {
        'train': 799, 
        'val': 89, 
        'trainval': 888,
    },
    num_classes=2,
    ignore_label=255,
)

_COCO2017_HUMAN = DatasetDescriptor(
    splits_to_sizes = {
        'train': 57704,
        'val': 6411,
        'trainval': 64115,
    },
    num_classes=2,
    ignore_label=255,
)

_COCO2017_SALIENCY = DatasetDescriptor(
    splits_to_sizes = {
        'train': 75287,
        'val': 8375
    },
    num_classes=2,
    ignore_label=255,
)

_COCO2017_SALIENCY_EXT = DatasetDescriptor(
    splits_to_sizes = {
        'train': 82408,
        'val': 2556,
    },
    num_classes=2,
    ignore_label=255,
)

_PASCAL_VOC_EXT = DatasetDescriptor(
    splits_to_sizes = {
        'train': 8497,
        'val': 2857,
    },
    num_classes=2,
    ignore_label=255,
)

_PASCAL_VOC_SALIENCY = DatasetDescriptor(
    splits_to_sizes = {
        'train': 1733,
        'val': 194,
    },
    num_classes=2,
    ignore_label=255,
)


_TEST_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'train': 1,
        'val': 1,
    },
    num_classes=2,
    ignore_label=255,
)

_COCO_INSTANCE_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'coco_train_instance':209988,
        'coco_test_instance':52477,
    },
    num_classes=2,
    ignore_label=255,
)

_COCO_INSTANCE_SIMPLE_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'coco_train_instance':121759,
        'coco_test_instance':6403,
    },
    num_classes=2,
    ignore_label=255,
)

_LV_INSTANCE_SIMPLE_INFORMATION = DatasetDescriptor(
    splits_to_sizes = {
        'LV-train':14226,
        'LV-test':736,
    },
    num_classes=2,
    ignore_label=255,
)
_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'owlii_studio': _OWLII_STUDIO_INFORMATION,
    'pascal_voc_human': _PASCAL_VOC_HUMAN,
    'pascal_voc_saliency': _PASCAL_VOC_SALIENCY,
    'pascal_voc_ext': _PASCAL_VOC_EXT,
    'coco2017_human': _COCO2017_HUMAN,
    'coco2017_saliency': _COCO2017_SALIENCY,
    'coco2017_saliency_ext': _COCO2017_SALIENCY_EXT,
    'test': _TEST_INFORMATION,
    'coco-instance': _COCO_INSTANCE_INFORMATION,
    'coco-instance-simple': _COCO_INSTANCE_SIMPLE_INFORMATION,
    'LV-instance-simple': _LV_INSTANCE_SIMPLE_INFORMATION
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def get_cityscapes_dataset_name():
  return 'cityscapes'


def get_dataset(dataset_name, split_name, dataset_dir):
  """Gets an instance of slim Dataset.

  Args:
    dataset_name: Dataset name.
    split_name: A train/val Split name.
    dataset_dir: The directory of the dataset sources.

  Returns:
    An instance of slim Dataset.

  Raises:
    ValueError: if the dataset_name or split_name is not recognized.
  """
  if dataset_name not in _DATASETS_INFORMATION:
    raise ValueError('The specified dataset is not supported yet.')

  splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

  if split_name not in splits_to_sizes:
    raise ValueError('data split name %s not recognized' % split_name)

  # Prepare the variables for different datasets.
  num_classes = _DATASETS_INFORMATION[dataset_name].num_classes
  ignore_label = _DATASETS_INFORMATION[dataset_name].ignore_label

  file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Specify how the TF-Examples are decoded.
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpg'),
      'image/height': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/segmentation/class/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/segmentation/class/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
  }
  items_to_handlers = {
      'image': tfexample_decoder.Image(
          image_key='image/encoded',
          format_key='image/format',
          channels=3),
      'image_name': tfexample_decoder.Tensor('image/filename'),
      'height': tfexample_decoder.Tensor('image/height'),
      'width': tfexample_decoder.Tensor('image/width'),
      'labels_class': tfexample_decoder.Image(
          image_key='image/segmentation/class/encoded',
          format_key='image/segmentation/class/format',
          channels=1),
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=splits_to_sizes[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      ignore_label=ignore_label,
      num_classes=num_classes,
      name=dataset_name,
      multi_label=True)
