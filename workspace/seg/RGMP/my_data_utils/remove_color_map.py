"""
Removes the color map from segmentation annotations.
Removes the color map from the ground truth segmentation annotations and save the results to output_dir.
"""

import glob
import os.path
import numpy as np
from PIL import Image
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('original_gt_folder', 
                           './VOCdevkit/VOC2012/SegmentationClass',
                           'Original ground truth annotations.')
tf.app.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')
tf.app.flags.DEFINE_string('output_dir',
                           './VOCdevkit/VOC2012/SegmentationClassRaw',
                           'folder to save modified ground truth annotations.')

def _remove_colormap(filename):
  return np.array(Image.open(filename))

def _save_annotation(annotation, filename):
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  with tf.gfile.Open(filename, mode='w') as f:
    pil_image.save(f, 'PNG')

def main(unused_argv):
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,
                                       '*.' + FLAGS.segmentation_format))

  for annotation in annotations:
    raw_annotation = _remove_colormap(annotation)
    filename = os.path.basename(annotation)[:-4]
    _save_annotation(raw_annotation,
                     os.path.join(FLAGS.output_dir, 
                         filename + '.' + FLAGS.segmentation_format))

if __name__ == '__main__':
  tf.app.run()






