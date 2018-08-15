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
model_dir = workspace + '/RGMP/modelvideoseq/ten/deploy/'
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
      graph_def = tf.GraphDef().FromString(f.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image, mask, first_feature):
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

    the_input = [np.concatenate((np.asarray(resized_image),mask),2)]

    t1 = time.time()
    heatmap = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: the_input,self.FIRST_FEATURE_NAME:first_feature})
    t2 = time.time()
    total = (t2 - t1) * 1000

    heatmap = heatmap[0, :, :, :]

    print('heatmap max:%d'%(heatmap.max()))

    # print(feature[0].max())
    # print(feature[0].min())

    return heatmap, total


  def first(self, image, mask):
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
    the_input = [np.concatenate((np.asarray(resized_image),mask),2)]


    t1 = time.time()
    feature_node = self.graph.get_operation_by_name(self.INTERMEDIATE_NAME).outputs[0]
    feature = self.sess.run(
        [feature_node],
        feed_dict={self.INPUT_TENSOR_NAME: the_input,self.FIRST_FEATURE_NAME:np.zeros((1,16,16,64))})
    t2 = time.time()
    total = (t2 - t1) * 1000

    feature = feature[0]

    return feature, total



def infer_and_store(filename, use_heatmap=True):
  # image preprocessing
#  img = Image.open(image_path)
  filelist = ls(img_dir+filename)
  filelist.sort()
  first_flag = False
  feature = []
  model = SgmtModel()
  total_time = 0
  cnt = 0
  mask = []
  saveroot = res_dir + filename + '/'
  if not os.path.exists(saveroot):
    os.makedirs(saveroot)
  for file in filelist:
    if file[-3:] != 'jpg':
      continue 
    image = Image.open(img_dir + filename + '/' + file)
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
    color = [127.5, 127.5, 127.5]
    img_array = np.array(image)
    img_array = cv2.copyMakeBorder(img_array, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    image = Image.fromarray(np.uint8(img_array))

  # run model
    if first_flag == False:
      mask = Image.open(mask_dir + filename + '/' + file[:-3] + 'png')
      mask = mask.resize((new_width, new_height), Image.ANTIALIAS)
      mask_array = np.array(mask)
      mask_array = cv2.copyMakeBorder(mask_array, top, bottom, left, right, cv2.BORDER_CONSTANT,
          value=[0])
      mask = mask_array[:,:,np.newaxis]

      
      first_flag = True
      feature, running_time = model.first(image,mask)
      total_time += running_time

      continue    
    
    heatmap, running_time = model.run(image,mask,feature)
    total_time += running_time
    cnt += 1
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
    mask = np.zeros((heatmap_array.shape[0],heatmap_array.shape[1]))
    mask[0:new_height,0:new_width] = heatmap_array[0:new_height, 0:new_width]
    mask = np.where(mask>0.5,1,0)
    mask = mask[:,:,np.newaxis]
    heatmap_array = heatmap_array[0:new_height, 0:new_width] * 255
    print('xxxxxx')
    print('result max:%d'%(heatmap_array.max()))
    heatmap_crop = Image.fromarray(np.uint8(heatmap_array))
#  heatmap_crop.save('data/heatmap_tf.png')


    embed_crop.save(saveroot + 'embed'+ file[:-3] + 'png')
    heatmap_crop.save(saveroot + 'heatmap'+ file[:-3] + 'png')

  if cnt == 0:
    return 0
  else:
    return float(total_time)/float(cnt)


# now start inferring
with open(listpath) as f:
    lines = f.readlines()
lines = [x.strip('\n') for x in lines] 
#print(lines)

for filename in lines:
  #image = Image.open(data_dir + filename)
  running_time = infer_and_store(filename)
  
  # image.save(res_dir + filename)
  # embed_crop.save(res_dir + filename_root + '.embed_tf.png')
  # heatmap_crop.save(res_dir + filename_root + '.heatmap_tf.png')
  print('Time consumed on ', filename, ': ', running_time, ' ms.')



