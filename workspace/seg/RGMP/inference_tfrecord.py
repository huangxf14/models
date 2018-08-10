import os
import sys
from io import BytesIO
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
import time

from os import listdir as ls


workspace = '/home/corp.owlii.com/xiufeng.huang/models/workspace/seg/'
data_dir = '/home/corp.owlii.com/xiufeng.huang/DAVIS/'
model_dir = workspace + '/RGMP/modelvideo_test/all/deploy/'
res_dir = model_dir + 'res/'
if not os.path.exists(res_dir):
  os.makedirs(res_dir)

img_dir = data_dir + 'JPEGImages/480p/'
mask_dir = data_dir + 'Annotations/480p-human/'
listpath = data_dir + 'ImageSets/2016/val-human-2016.txt'

FROZEN_GRAPH_NAME = model_dir + 'deploy_graph.pb'
INPUT_SIZE = 512

class SgmtModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'image:0'
  FIRST_FEATURE_NAME = 'first:0'
  OUTPUT_TENSOR_NAME = 'heatmap:0'
  INTERMEDIATE_NAME = 'MobilenetV2/Conv_1/Relu6'

  def __init__(self):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    graph_def = None
    with tf.gfile.GFile(FROZEN_GRAPH_NAME, "rb") as f:
      print(FROZEN_GRAPH_NAME)
      graph_def = tf.GraphDef().FromString(f.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    t1 = time.time()
    heatmap = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        )
    t2 = time.time()
    total = (t2 - t1) * 1000

    heatmap = heatmap[0, :, :, :]

    

    # print(feature[0].max())
    # print(feature[0].min())

    return heatmap, total






def infer_and_store():

  feature = []
  model = SgmtModel()

  print('run')

  heatmap, running_time = model.run()

  print(heatmap.shape)

  return running_time



running_time = infer_and_store()
  

print('Time consumed on ', filename, ': ', running_time, ' ms.')



