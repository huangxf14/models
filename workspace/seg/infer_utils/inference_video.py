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

model_dir = workspace + '/coco-instance/model075_256/all/deploy/'
res_dir = '/home/corp.owlii.com/xiufeng.huang/models/workspace/seg/infer_utils/res/'
if not os.path.exists(res_dir):
  os.makedirs(res_dir)

data_dir = '/home/corp.owlii.com/xiufeng.huang/DAVIS/JPEGImages/480p/dance-twirl/'
mask_dir = '/home/corp.owlii.com/xiufeng.huang/DAVIS/Annotations/480p/dance-twirl/' 
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
  # img = Image.open(image_path)
  width, height = image.shape[1],image.shape[0]
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
  
  image = cv2.resize(image,(new_width,new_height))
  
  # padding
  delta_w = INPUT_SIZE - new_width
  delta_h = INPUT_SIZE - new_height
  top, bottom = 0, delta_h
  left, right = 0, delta_w
  color = [127, 127, 127]
  img_array = image #np.array(image)
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
  embed_crop = cv2.resize(embed_crop,(width,height))
  embed_crop = np.uint8(embed_crop)
  #embed_crop = Image.fromarray(np.uint8(embed_crop))
  # embed_crop.save('data/embed_tf.png')
  
  heatmap_array = np.squeeze(heatmap)
  heatmap_array = heatmap_array[0:new_height, 0:new_width] * 255
  heatmap_crop = cv2.resize(heatmap_array,(width,height))
  #heatmap_crop = Image.fromarrsay(np.uint8(heatmap_array))
  heatmap_crop = np.uint8(heatmap_crop)
  # heatmap_crop.save('data/heatmap_tf.png')

  return embed_crop, heatmap_crop, running_time



# now start inferring
# with open(listpath) as f:
#     lines = f.readlines()
# lines = [x.strip('\n') for x in lines] 
#print(lines)
lines = ls(data_dir)

lines.sort()

bbox = None
bbox_tmp = None

for filename in lines:
  filename_root = os.path.splitext(filename)[0]

  if bbox is None:
    mask = cv2.imread(mask_dir + filename[:-3]+'png',cv2.IMREAD_GRAYSCALE)
    bbox = [mask.shape[0],mask.shape[1],0,0]
    for x in range(mask.shape[0]):
      for y in range(mask.shape[1]):
        if mask[x][y] > 0:
          if bbox[0] > x:
            bbox[0] = x
          if bbox[1] > y:
            bbox[1] = y
          if bbox[2] < x:
            bbox[2] = x
          if bbox[3] < y:
            bbox[3] = y
    continue

  image_raw = cv2.imread(data_dir + filename)
  if bbox is not None:
    print(bbox)
    image = image_raw[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1,:]

  embed_crop, heatmap_crop, running_time = infer_one(image)
  embed = np.zeros((image_raw.shape[0],image_raw.shape[1],image_raw.shape[2]),dtype=np.uint8)
  embed[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1,:] = embed_crop
  heatmap = np.zeros((image_raw.shape[0],image_raw.shape[1]),dtype=np.uint8)
  heatmap[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1] = heatmap_crop

  print('xxxxxxxxxxx')
  print(filename)
  print(heatmap_crop.shape)
  print(heatmap_crop.max())
  bbox_tmp = [heatmap_crop.shape[0],heatmap_crop.shape[1],0,0]
  for x  in range(heatmap_crop.shape[0]):
    for y in range(heatmap_crop.shape[1]):
      if heatmap_crop[x][y] > 127:
        if bbox_tmp[0] > x:
          bbox_tmp[0] = x
        if bbox_tmp[1] > y:
          bbox_tmp[1] = y
        if bbox_tmp[2] < x:
          bbox_tmp[2] = x
        if bbox_tmp[3] < y:
          bbox_tmp[3] = y
  for cnt in range(3,-1,-1):
    bbox[cnt] = bbox[cnt%2]+bbox_tmp[cnt]
  
  print(bbox_tmp)
  print(bbox)

  dx = bbox[2] - bbox[0] + 1
  dy = bbox[3] - bbox[1] + 1
  bbox[0] -= dx * 0.25
  bbox[1] -= dy * 0.25
  bbox[2] += dx * 0.25
  bbox[3] += dy * 0.25
  bbox[0] = int(bbox[0])
  bbox[1] = int(bbox[1])
  bbox[2] = int(bbox[2])
  bbox[3] = int(bbox[3])
  bbox[0] = max(bbox[0],0)
  bbox[1] = max(bbox[1],0)
  bbox[2] = min(bbox[2],image_raw.shape[0]-1)
  bbox[3] = min(bbox[3],image_raw.shape[1]-1)

  
  cv2.imwrite(res_dir + filename,image_raw)

  cv2.imwrite(res_dir + filename_root + '.embed_tf.png',embed)
  cv2.imwrite(res_dir + filename_root + '.heatmap_tf.png',heatmap)
  print('Time consumed on ', filename, ': ', running_time, ' ms.')




