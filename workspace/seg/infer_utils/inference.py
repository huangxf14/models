import os
import sys
from io import BytesIO
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
import time


workspace = '/home/corp.owlii.com/xiufeng.huang/models/workspace/seg/'

model_dir = workspace + '/LV-instance1/model035_instance/all/deploy/'
res_dir = model_dir + 'res/'
if not os.path.exists(res_dir):
  os.makedirs(res_dir)

data_dir = workspace + 'data/'
listpath = data_dir + 'list.txt'

FROZEN_GRAPH_NAME = model_dir + 'deploy_graph.pb'
INPUT_SIZE = 512

class SgmtModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'image:0'
  OUTPUT_TENSOR_NAME = 'heatmap:0'

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

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    the_input = [np.asarray(resized_image)]

    t1 = time.time()
    heatmap = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: the_input})
    t2 = time.time()
    total = (t2 - t1) * 1000

    heatmap = heatmap[0, :, :, :]

    return heatmap, total


def infer_one(image, use_heatmap=True):
  # image preprocessing
#  img = Image.open(image_path)
  width, height = image.size
  large_one = max(width, height)
  
  scale = float(INPUT_SIZE) / float(large_one)
  
  new_width = 0
  new_height = 0
  if width >= height:
    new_width = INPUT_SIZE
    new_height = int(height * scale)
  else:
    new_height = INPUT_SIZE
    new_width = int(width * scale)
  
  image = image.resize((new_width, new_height), Image.ANTIALIAS)
  
  # padding
  delta_w = INPUT_SIZE - new_width
  delta_h = INPUT_SIZE - new_height
  top, bottom = 0, delta_h
  left, right = 0, delta_w
  color = [127, 127, 127]
  img_array = np.array(image)
  img_array = cv2.copyMakeBorder(img_array, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=color)
  
  image = Image.fromarray(np.uint8(img_array))
  
  # run model
  model = SgmtModel()
  heatmap, running_time = model.run(image)
  heatmap = np.float32(heatmap) / 255.0
  if not use_heatmap:
    heatmap = np.where(heatmap > 0.5, 1, 0)

  # post processing
  embed_array = img_array
  embed_array = np.multiply(img_array, heatmap) 
  
  # get results
  embed_crop = embed_array[0:new_height, 0:new_width]
  embed_crop = Image.fromarray(np.uint8(embed_crop))
#  embed_crop.save('data/embed_tf.png')
  
  heatmap_array = np.squeeze(heatmap)
  heatmap_array = heatmap_array[0:new_height, 0:new_width] * 255
  heatmap_crop = Image.fromarray(np.uint8(heatmap_array))
#  heatmap_crop.save('data/heatmap_tf.png')

  return embed_crop, heatmap_crop, running_time



# now start inferring
with open(listpath) as f:
    lines = f.readlines()
lines = [x.strip('\n') for x in lines] 
#print(lines)

for filename in lines:
  filename_root = os.path.splitext(filename)[0]

  image = Image.open(data_dir + filename)
  embed_crop, heatmap_crop, running_time = infer_one(image)
  
  image.save(res_dir + filename)
  embed_crop.save(res_dir + filename_root + '.embed_tf.png')
  heatmap_crop.save(res_dir + filename_root + '.heatmap_tf.png')
  print('Time consumed on ', filename, ': ', running_time, ' ms.')



