import tensorflow as tf
import tfcoreml as tf_converter

tf.app.flags.DEFINE_string(
    'input_pb_file',
    '/home/corp.owlii.com/yi.xu/workspace/sgmt/train/deploy/deploy_graph.pb',
    'Input tensorflow pb file')

tf.app.flags.DEFINE_string(
    'output_mlmodel',
    '/home/corp.owlii.com/yi.xu/workspace/sgmt/train/deploy/deploy_graph.mlmodel',
    'Output coreml model file')

tf.app.flags.DEFINE_string(
    'input_node_name',
    'image:0',
    'As mentioned')

tf.app.flags.DEFINE_string(
    'output_node_name',
    'MobilenetV2/ResizeBilinear:0',
    'As mentioned')

tf.app.flags.DEFINE_string(
    'input_shape', '1,512,512,3', 'As mentioned')

FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
  input_shape = [int(n) for n in FLAGS.input_shape.split(',')]
  tf_converter.convert(tf_model_path = FLAGS.input_pb_file,
                       mlmodel_path = FLAGS.output_mlmodel,
                       output_feature_names = [FLAGS.output_node_name],
                       image_input_names = [FLAGS.input_node_name],
                       input_name_shape_dict = 
                           {FLAGS.input_node_name: input_shape})

#                       input_name_shape_dict = {FLAGS.input_node_name : FLAGS.input_size})
#                       red_bias = -123.68,
#                       green_bias = -116.78, 
#                       blue_bias = -103.94)
#                       image_scale = 2.0/255.0)


if __name__ == '__main__':
  tf.app.run()

