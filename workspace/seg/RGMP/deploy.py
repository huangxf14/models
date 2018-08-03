# Adjusted form models/research/object_detection/export.py
# ==============================================================================

"""Functions to export object detection inference graph."""
import logging
import os
import tensorflow as tf
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as saver_lib
from net import nets_factory


flags = tf.app.flags

flags.DEFINE_string('input_shape',
                    '1,512,512,4', 'As mentioned')
flags.DEFINE_string('first_shape',
                    '1,16,16,64', 'As mentioned')
flags.DEFINE_string('output_size',
                    '512,512', 'As mentioned')
flags.DEFINE_integer('num_classes', 2, 'As mentioned')
flags.DEFINE_string('trained_checkpoint_prefix', None,
                    'Path to trained ckpt')
flags.DEFINE_string('output_directory', None, 
                    'Path to write outputs.')
flags.DEFINE_string('output_graph_name', 
                    'frozen_inference_graph.pb', 
                    'As mentioned.')
flags.DEFINE_string(
    'model_name', None, 'As mentioned')
flags.DEFINE_bool('deploy_output_resizer', True, 'As mentioned')

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS



def freeze_graph_with_def_protos(
    input_graph_def,
    input_saver_def,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    clear_devices,
    initializer_nodes,
    variable_names_blacklist=''):
  """Converts all variables in a graph and checkpoint into constants."""
  del restore_op_name, filename_tensor_name  # Unused by updated loading code.

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if not saver_lib.checkpoint_exists(input_checkpoint):
    raise ValueError(
        'Input checkpoint "' + input_checkpoint + '" does not exist!')

  if not output_node_names:
    raise ValueError(
        'You must supply the name of a node to --output_node_names.')

  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ''

  with tf.Graph().as_default():
    tf.import_graph_def(input_graph_def, name='')
    config = tf.ConfigProto(graph_options=tf.GraphOptions())
#    with session.Session(config=config) as sess:
    with session.Session() as sess:
      saver = saver_lib.Saver(saver_def=input_saver_def)
      saver.restore(sess, input_checkpoint)
      variable_names_blacklist = (variable_names_blacklist.split(',') if
                                  variable_names_blacklist else None)

      output_graph_def = graph_util.convert_variables_to_constants(
          sess,
          input_graph_def,
          output_node_names.split(','),
          variable_names_blacklist=variable_names_blacklist)

  return output_graph_def


def _add_output_for_deploy(input_tensors,
                           output_tensors,
                           output_collection_name='inference_op'):

  outputs = {}
  outputs['heatmap'] = output_tensors
  tf.add_to_collection(output_collection_name, outputs['heatmap']) # what's the point of it?
  return outputs


def write_frozen_graph(frozen_graph_path, frozen_graph_def):
  """Writes frozen graph to disk.
  Args:
    frozen_graph_path: Path to write inference graph.
    frozen_graph_def: tf.GraphDef holding frozen graph.
  """
  with gfile.GFile(frozen_graph_path, 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())
  logging.info('%d ops in the final graph.', len(frozen_graph_def.node))


def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
  """Writes the graph and the checkpoint into disk."""
  for node in inference_graph_def.node:
    node.device = ''
  with tf.Graph().as_default():
    tf.import_graph_def(inference_graph_def, name='')
    with session.Session() as sess:
      saver = saver_lib.Saver(saver_def=input_saver_def,
                              save_relative_paths=True)
      saver.restore(sess, trained_checkpoint_prefix)
      saver.save(sess, model_path)


def _get_outputs_from_inputs(input_tensors, 
                             first_feature,
                             network_fn,
                             output_collection_name):

  logits, end_points = network_fn(input_tensors,first_feature)
  heatmap = tf.nn.softmax(logits)

  # try slicing heatmap by multiplying, and convert to uint8 in [0, 255]
#  heatmap = tf.slice(heatmap, [0,0,0,0], [1,512,512,1])
  slicing_multiplier = tf.constant([0.0, 255.0], dtype=tf.float32)
  heatmap = tf.reduce_max(heatmap * slicing_multiplier, 
                          axis=3, keepdims=True)
  heatmap = tf.cast(heatmap, tf.uint8)

  # resize, if necessary
  output_size = [int(n) for n in FLAGS.output_size.split(',')]
  if not FLAGS.deploy_output_resizer:
    output_tensors = tf.identity(heatmap, name='heatmap')
  else:
    output_tensors = tf.image.resize_bilinear(
        heatmap, output_size, align_corners=True, name='resize_to_heatmap')
    output_tensors = tf.identity(output_tensors, name='heatmap')

  return _add_output_for_deploy(input_tensors, output_tensors,
                                output_collection_name)

def _build_graph(network_fn, 
                 input_shape,
                 output_collection_name):

  placeholder_tensor = tf.placeholder(
      dtype=tf.uint8, shape=input_shape, name='image')

  first_shape = [int(n) for n in FLAGS.first_shape.split(',')]
  placeholder_first = tf.placeholder(
      dtype=tf.float32, shape=first_shape, name='first')

  placeholder_tensor = tf.to_float(placeholder_tensor)

  outputs = _get_outputs_from_inputs(
      input_tensors=placeholder_tensor,
      first_feature=placeholder_first,
      network_fn=network_fn,
      output_collection_name=output_collection_name)

  # Add global step to the graph.
  slim.get_or_create_global_step()

  return outputs, placeholder_tensor


def _export_inference_graph(network_fn,
                            trained_checkpoint_prefix,
                            output_directory,
                            input_shape=None,
                            output_collection_name='inference_op'):
  tf.gfile.MakeDirs(output_directory)
  frozen_graph_path = os.path.join(output_directory,
                                   FLAGS.output_graph_name)

  ckpt_path = os.path.join(output_directory, 'model.ckpt')

  outputs, placeholder_tensor = _build_graph(
      network_fn=network_fn,
      input_shape=input_shape,
      output_collection_name=output_collection_name)

  saver_kwargs = {}
  checkpoint_to_use = trained_checkpoint_prefix

  saver = tf.train.Saver(**saver_kwargs)
  input_saver_def = saver.as_saver_def()

  write_graph_and_checkpoint(
      inference_graph_def=tf.get_default_graph().as_graph_def(),
      model_path=ckpt_path,
      input_saver_def=input_saver_def,
      trained_checkpoint_prefix=checkpoint_to_use)

  output_node_names = ','.join(outputs.keys())

  frozen_graph_def = freeze_graph_with_def_protos(
      input_graph_def=tf.get_default_graph().as_graph_def(),
      input_saver_def=input_saver_def,
      input_checkpoint=checkpoint_to_use,
      output_node_names=output_node_names,
      restore_op_name='save/restore_all',
      filename_tensor_name='save/Const:0',
      clear_devices=True,
      initializer_nodes='')
  write_frozen_graph(frozen_graph_path, frozen_graph_def)

def export_inference_graph(trained_checkpoint_prefix,
                           output_directory,
                           input_shape = [1, 512, 512, 3],
                           num_classes = 2,
                           output_collection_name='inference_op',
                           additional_output_tensor_names=None):

  weight_decay = 0.00004
  network_fn = nets_factory.get_network_fn(
      FLAGS.model_name,
      num_classes=num_classes,
      weight_decay=weight_decay,
      is_training=False)

  _export_inference_graph(
      network_fn,
      trained_checkpoint_prefix,
      output_directory,
      input_shape,
      output_collection_name)


def main(_):
  input_shape = [int(n) for n in FLAGS.input_shape.split(',')]
  print('### deploy.py ### - deploy input shape: ', input_shape)
  export_inference_graph(FLAGS.trained_checkpoint_prefix,
                         FLAGS.output_directory, 
                         input_shape, 
                         FLAGS.num_classes)

if __name__ == '__main__':
  tf.app.run()
