from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import deeplab
from deeplab import common
#from deeplab.datasets import segmentation_dataset
from my_data_utils import input_generator_video
from deeplab.utils import train_utils
from datasets import dataset_factory
from deployment import model_deploy
#from nets import nets_factory
from preprocessing import preprocessing_factory

from net import nets_factory
from my_data_utils import segmentation_dataset
slim = tf.contrib.slim

dataset_name = 'DAVIS-video'
dataset_split_name = 'test'
dataset_dir = '/home/corp.owlii.com/xiufeng.huang/models/workspace/seg/RGMP/tfrecord-video'
train_crop_size = [512,512]
clone_batch_size = 1

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')


FLAGS = tf.app.flags.FLAGS

# def main(_):
def get_network_fn():
    dataset = segmentation_dataset.get_video_dataset(
            dataset_name, dataset_split_name, dataset_dir=dataset_dir)

    samples = input_generator_video.get(
                dataset,
                train_crop_size,
                clone_batch_size,
                min_resize_value=None,
                max_resize_value=None,
                resize_factor=None,
                min_scale_factor=0.5,
                max_scale_factor=2.0,
                scale_factor_step_size=0,
                dataset_split=dataset_split_name,
                is_training=False,
                model_variant=FLAGS.model_variant,
                test=True)

    return tf.identity(samples[common.IMAGE],name='input')


# if __name__ == '__main__':
#   tf.app.run()