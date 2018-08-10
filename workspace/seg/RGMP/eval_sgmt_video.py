"""
Evaluation script adjusted from slim.eval_image_classifier.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import math
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deeplab import common
from deeplab import model
#from deeplab.datasets import segmentation_dataset
from my_data_utils import input_generator_video
from datasets import dataset_factory
#from nets import nets_factory
from preprocessing import preprocessing_factory

import sys
sys.path.append('..')
from my_data_utils import segmentation_dataset

from net import nets_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'pascal_voc_seg', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'val', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v2_sgmt', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#tf.app.flags.DEFINE_integer(
#    'eval_image_size', None, 'Eval image size')

#tf.app.flags.DEFINE_integer(
#    'eval_batch_size', 1,
#    'The number of images in each batch during evaluation.')

tf.app.flags.DEFINE_multi_integer(
    'eval_crop_size', [512, 512],
    'Image crop size [height, width] for evaluation.')

#tf.app.flags.DEFINE_multi_integer(
#    'output_size', [512, 512], 'As mentioned')

tf.app.flags.DEFINE_integer(
    'min_resize_value', 512, 'As mentioned')

tf.app.flags.DEFINE_integer(
    'max_resize_value', 512, 'As mentioned')

tf.app.flags.DEFINE_float(
    'min_scale_factor', 1.0, 'As mentioned')

tf.app.flags.DEFINE_float(
    'max_scale_factor', 1.0, 'As mentioned')

#tf.app.flags.DEFINE_float(
#    'resize_factor', None,
#    'As mentioned.')

tf.app.flags.DEFINE_string(
    'eval_split', 'val',
    'Which split of the dataset used for evaluation')

tf.app.flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

#tf.app.flags.DEFINE_boolean(
#    'adjust_for_deploy', True, 'As mentioned')

tf.app.flags.DEFINE_integer('num_clones', 1, 'As mentioned')

tf.app.flags.DEFINE_bool('use_cpu', False, 'As mentioned')


#####################
# instance seg flags #
#####################
tf.app.flags.DEFINE_boolean(
    'instance_seg', False, 'As mentioned')
tf.app.flags.DEFINE_float(
    'inner_extension_ratio', -0.1, 'As mentioned')
tf.app.flags.DEFINE_float(
    'outer_extension_ratio', 0.2, 'As mentioned')
tf.app.flags.DEFINE_string(
    'filling', 'central_padding', 'central_padding or resize')


FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
#    dataset = dataset_factory.get_dataset(
#        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    dataset = segmentation_dataset.get_video10_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, dataset_dir=FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
#    provider = slim.dataset_data_provider.DatasetDataProvider(
#        dataset,
#        shuffle=False,
#        common_queue_capacity=2 * FLAGS.batch_size,
#        common_queue_min=FLAGS.batch_size)
#    [image, label] = provider.get(['image', 'label'])
#    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
#    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
#    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
#        preprocessing_name,
#        is_training=False)

#    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
#    eval_image_size = 224

#    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

#    images, labels = tf.train.batch(
#        [image, label],
#        batch_size=FLAGS.batch_size,
#        num_threads=FLAGS.num_preprocessing_threads,
#        capacity=5 * FLAGS.batch_size)


    if FLAGS.instance_seg:
      instance_seg_args = {
          'inner_extension_ratio': FLAGS.inner_extension_ratio,
          'outer_extension_ratio': FLAGS.outer_extension_ratio,
          'filling': FLAGS.filling
      }
      samples = input_generator.get(
          dataset, 
          FLAGS.eval_crop_size,
          FLAGS.batch_size,
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          min_scale_factor=FLAGS.min_scale_factor,
          max_scale_factor=FLAGS.max_scale_factor,
          dataset_split=FLAGS.eval_split,
          is_training=False,
          model_variant=FLAGS.model_variant,
          instance_seg=FLAGS.instance_seg,
          instance_seg_args=instance_seg_args)
    else:
      samples = input_generator_video.get(
          dataset, 
          FLAGS.eval_crop_size,
          FLAGS.batch_size,
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          min_scale_factor=FLAGS.min_scale_factor,
          max_scale_factor=FLAGS.max_scale_factor,
          dataset_split=FLAGS.eval_split,
          is_training=False,
          model_variant=FLAGS.model_variant)

    images = samples[common.IMAGE]
    labels = samples[common.LABEL]
    labels = labels[:,:,:,2:3]

    # make predictions!
    logits, end_points = network_fn(images)

    print('images.shape: ', images.shape)
    print('labels.shape: ', labels.shape)
    print('logits.shape: ', logits.shape)

    logits = tf.image.resize_bilinear(
        logits, tf.shape(labels)[1:3], align_corners=True)
    print('upsampled logits shape: ', logits.shape)


    variables_to_restore = slim.get_variables_to_restore()


    predictions = tf.argmax(tf.nn.softmax(logits), 3)
    # or predictions = tf.argmax(logits, 3) ?

    # flattens into 1-D
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(labels, shape=[-1])
    print('predictions.shape: ', predictions.shape)
    print('labels.shape: ', labels.shape)

    # make of vector of float such that weight[i] = 0.f if labels[i] = ignore_label, 
    # and weight[i] = 1.f if labels[i] != ignore_label.
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)
    print('adjusted labels.shape: ', labels.shape)

    predictions_tag = 'miou'

    # Define the evaluation metric.
    metric_map = {}
    metric_map[predictions_tag] = tf.metrics.mean_iou(
        predictions, labels, dataset.num_classes, weights=weights)

    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))

    for metric_name, metric_value in six.iteritems(metrics_to_values):
      slim.summaries.add_scalar_summary(
          metric_value, metric_name, print_summary=True)

    num_batches = int(
        math.ceil(dataset.num_samples / float(FLAGS.batch_size)))

    tf.logging.info('Eval num images %d', dataset.num_samples)
    tf.logging.info('Eval batch size %d and num batch %d',
                    FLAGS.batch_size, num_batches)

    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations

    if FLAGS.use_cpu:
      config = tf.ConfigProto(device_count={'GPU':0})
    else:
      config = None
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        max_number_of_evaluations=num_eval_iters,
        eval_interval_secs=FLAGS.eval_interval_secs,
        session_config=config)


    ####################
    # Define the model #
    ####################
#    logits, _ = network_fn(images)
#
#    if FLAGS.moving_average_decay:
#      variable_averages = tf.train.ExponentialMovingAverage(
#          FLAGS.moving_average_decay, tf_global_step)
#      variables_to_restore = variable_averages.variables_to_restore(
#          slim.get_model_variables())
#      variables_to_restore[tf_global_step.op.name] = tf_global_step
#    else:
#      variables_to_restore = slim.get_variables_to_restore()
#
#    predictions = tf.argmax(logits, 1)
#    labels = tf.squeeze(labels)
#
#    # Define the metrics:
#    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
#        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
#        'Recall_5': slim.metrics.streaming_recall_at_k(
#            logits, labels, 5),
#    })
#
#    # Print the summaries to screen.
#    for name, value in names_to_values.items():
#      summary_name = 'eval/%s' % name
#      op = tf.summary.scalar(summary_name, value, collections=[])
#      op = tf.Print(op, [value], summary_name)
#      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
#
#    # TODO(sguada) use num_epochs=1
#    if FLAGS.max_num_batches:
#      num_batches = FLAGS.max_num_batches
#    else:
#      # This ensures that we make a single pass over all of the data.
#      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
#
#    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
#      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
#    else:
#      checkpoint_path = FLAGS.checkpoint_path
#
#    tf.logging.info('Evaluating %s' % checkpoint_path)
#
#    slim.evaluation.evaluate_once(
#        master=FLAGS.master,
#        checkpoint_path=checkpoint_path,
#        logdir=FLAGS.eval_dir,
#        num_evals=num_batches,
#        eval_op=list(names_to_updates.values()),
#        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
