"""
Mobilenet Base Class for semantic segmentation.
Adjusted form slim/nets/mobilenet/mobilenet.py
Also, dropout layer is removed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import contextlib
import copy
import os

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_bool('use_decoder', False,
#                          'Whether to use decoder.')


@slim.add_arg_scope
def apply_activation(x, name=None, activation_fn=None):
  return activation_fn(x, name=name) if activation_fn else x


def _fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  Pads the input such that if it was used in a convolution with 'VALID' padding,
  the output would have the same dimensions as if the unpadded input was used
  in a convolution with 'SAME' padding.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                           kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
  pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
  pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
  pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                  [pad_beg[1], pad_end[1]], [0, 0]])
  return padded_inputs


def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


@contextlib.contextmanager
def _set_arg_scope_defaults(defaults):
  """Sets arg scope defaults for all items present in defaults.

  Args:
    defaults: dictionary/list of pairs, containing a mapping from
    function to a dictionary of default args.

  Yields:
    context manager where all defaults are set.
  """
  if hasattr(defaults, 'items'):
    items = list(defaults.items())
  else:
    items = defaults
  if not items:
    yield
  else:
    func, default_arg = items[0]
    with slim.arg_scope(func, **default_arg):
      with _set_arg_scope_defaults(items[1:]):
        yield


@slim.add_arg_scope
def depth_multiplier(output_params,
                     multiplier,
                     divisible_by=2,
                     min_depth=2,
                     **unused_kwargs):
  if 'num_outputs' not in output_params:
    return
  d = output_params['num_outputs']
  output_params['num_outputs'] = _make_divisible(d * multiplier, divisible_by,
                                                 min_depth)


_Op = collections.namedtuple('Op', ['op', 'params', 'multiplier_func'])


def op(opfunc, **params):
  multiplier = params.pop('multiplier_transorm', depth_multiplier)
  return _Op(opfunc, params=params, multiplier_func=multiplier)


@slim.add_arg_scope
def mobilenet_base(  # pylint: disable=invalid-name
    inputs,
    conv_defs,
    multiplier=1.0,
    final_endpoint=None,
    output_stride=None,
    use_explicit_padding=False,
    scope=None,
    is_training=False):
  """Mobilenet base network.

  Constructs a network from inputs to the given final endpoint. By default
  the network is constructed in inference mode. To create network
  in training mode use:

  with slim.arg_scope(mobilenet.training_scope()):
     logits, endpoints = mobilenet_base(...)

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    conv_defs: A list of op(...) layers specifying the net architecture.
    multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    final_endpoint: The name of last layer, for early termination for
    for V1-based networks: last layer is "layer_14", for V2: "layer_20"
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 1 or any even number, excluding
      zero. Typical values are 8 (accurate fully convolutional mode), 16
      (fast fully convolutional mode), and 32 (classification mode).

      NOTE- output_stride relies on all consequent operators to support dilated
      operators via "rate" parameter. This might require wrapping non-conv
      operators to operate properly.

    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.
    scope: optional variable scope.
    is_training: How to setup batch_norm and other ops. Note: most of the time
      this does not need be set directly. Use mobilenet.training_scope() to set
      up training instead. This parameter is here for backward compatibility
      only. It is safe to set it to the value matching
      training_scope(is_training=...). It is also safe to explicitly set
      it to False, even if there is outer training_scope set to to training.
      (The network will be built in inference mode).
  Returns:
    tensor_out: output tensor.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: depth_multiplier <= 0, or the target output_stride is not
                allowed.
  """
  if multiplier <= 0:
    raise ValueError('multiplier is not greater than zero.')

  # Set conv defs defaults and overrides.
  conv_defs_defaults = conv_defs.get('defaults', {})
  conv_defs_overrides = conv_defs.get('overrides', {})
  if use_explicit_padding:
    conv_defs_overrides = copy.deepcopy(conv_defs_overrides)
    conv_defs_overrides[
        (slim.conv2d, slim.separable_conv2d)] = {'padding': 'VALID'}

  if output_stride is not None:
    if output_stride == 0 or (output_stride > 1 and output_stride % 2):
      raise ValueError('Output stride must be None, 1 or a multiple of 2.')

  # a) Set the tensorflow scope
  # b) set padding to default: note we might consider removing this
  # since it is also set by mobilenet_scope
  # c) set all defaults
  # d) set all extra overrides.
  with _scope_all(scope, default_scope='Mobilenet'), \
      slim.arg_scope([slim.batch_norm], is_training=is_training), \
      _set_arg_scope_defaults(conv_defs_defaults), \
      _set_arg_scope_defaults(conv_defs_overrides):
    # The current_stride variable keeps track of the output stride of the
    # activations, i.e., the running product of convolution strides up to the
    # current network layer. This allows us to invoke atrous convolution
    # whenever applying the next convolution would result in the activations
    # having output stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    net = inputs
    # Insert default parameters before the base scope which includes
    # any custom overrides set in mobilenet.
    end_points = {}
    scopes = {}

    for i, opdef in enumerate(conv_defs['spec']):
#      print('####################: ', opdef)
      params = dict(opdef.params)
      opdef.multiplier_func(params, multiplier)
      stride = params.get('stride', 1)
      if output_stride is not None and current_stride == output_stride:
        # If we have reached the target output_stride, then we need to employ
        # atrous convolution with stride=1 and multiply the atrous rate by the
        # current unit's stride for use in subsequent layers.
        layer_stride = 1
        layer_rate = rate
        rate *= stride
      else:
        layer_stride = stride
        layer_rate = 1
        current_stride *= stride
      # Update params.
      params['stride'] = layer_stride
      # Only insert rate to params if rate > 1.
      if layer_rate > 1:
        params['rate'] = layer_rate
      # Set padding
      if use_explicit_padding:
        if 'kernel_size' in params:
          net = _fixed_padding(net, params['kernel_size'], layer_rate)
        else:
          params['use_explicit_padding'] = True

      end_point = 'layer_%d' % (i + 1)
      try:
        net = opdef.op(net, **params)
      except Exception:
        print('Failed to create op %i: %r params: %r' % (i, opdef, params))
        raise
      end_points[end_point] = net
      scope = os.path.dirname(net.name)
      scopes[scope] = end_point
      if final_endpoint is not None and end_point == final_endpoint:
        break

    # Add all tensors that end with 'output' to
    # endpoints
    for t in net.graph.get_operations():
      scope = os.path.dirname(t.name)
      bn = os.path.basename(t.name)
      if scope in scopes and t.name.endswith('output'):
        end_points[scopes[scope] + '/' + bn] = t.outputs[0]
    return net, end_points


@contextlib.contextmanager
def _scope_all(scope, default_scope=None):
  with tf.variable_scope(scope, default_name=default_scope) as s,\
       tf.name_scope(s.original_name_scope):
    yield s



@slim.add_arg_scope
def decoder(net_outputs, decoder_inputs, end_points,
            num_skips=5, skip_res=[[32, 32], [64, 64],
                [128, 128], [256, 256], [512, 512]],
            bridge_ch_nums=[8, 8, 8, 8, 8],
            output_ch_nums=[8, 8, 8, 8, 8],
            is_training=True,
            scope='Decoder'):
  # now we do the decoder here
  # this should be something right after the 1/8 embedding and before the logits
  # the original resolution input is "inputs" being [batch_size, 512, 512, 3]
  # the 1/2 input is end_points['layer_2'] being [batch_size, 256, 256, 8]
  # the 1/4 input is end_points['layer_4'] being [batch_size, 128, 128, 8]
  # while the embedding is "net" being [batch_size, 64, 64, 32]
  def bridge_conv_unit(inputs, num_outputs, scope=None):
    return slim.conv2d(
               inputs,
               num_outputs,
               [3, 3],
               scope=scope,
               weights_initializer=init_ops.zeros_initializer(),
#               weights_initializer=initializers.xavier_initializer(),
#               trainable=False,
               # activation_fn=tf.nn.relu,
               activation_fn=None,
               normalizer_fn=slim.batch_norm,
               normalizer_params={'center': True, 'scale': True,
                                  'is_training': is_training},
               biases_initializer=init_ops.zeros_initializer())

  def bridge_conv(inputs, num_outputs, scope=None):
    with tf.variable_scope(scope):
      net = bridge_conv_unit(inputs,num_outputs)
      net_res = bridge_conv_unit(net,num_outputs)
      net_res = bridge_conv_unit(net_res,num_outputs)
      return tf.identity(net+net_res,name='low_level_feature')

  def output_conv_unit(inputs, num_outputs, scope=None):
    return slim.conv2d(
               inputs,
               num_outputs,
               [3, 3],
               scope=scope,
               weights_initializer=initializers.xavier_initializer(),
               # activation_fn=tf.nn.relu6,
               activation_fn=None,
               normalizer_fn=slim.batch_norm,
               normalizer_params={'center': True, 'scale': True,
                                  'is_training': is_training},
               biases_initializer=init_ops.zeros_initializer())

  def output_conv(inputs, num_outputs, scope=None):
    with tf.variable_scope(scope):
      net = inputs
      net_res = output_conv_unit(net,num_outputs)
      net_res = output_conv_unit(net_res,num_outputs)
      return tf.identity(net+net_res,name='low_level_feature')


  temp_skip_res = [[64, 64], [128, 128], [256, 256], [512, 512]]
  with tf.variable_scope(scope):
    output = net_outputs
    for i in range(num_skips):
      bridge = decoder_inputs[i]
      if bridge_ch_nums[i] > 0:
        scope_name = 'skip_' + str(i+1) + '_' + str(temp_skip_res[i][0]) + 'x' + \
                         str(temp_skip_res[i][1])  + '_bridge'
        bridge = bridge_conv(bridge,
                   bridge_ch_nums[i],
                   scope=scope_name)
        end_points[scope_name] = bridge
      output = tf.image.resize_bilinear(
                   output, skip_res[i])
      # output = tf.concat([bridge, output], 3)
      output = bridge + output
      if output_ch_nums[i] > 0:
        scope_name = 'skip_' + str(i+1) + '_' + str(temp_skip_res[i][0]) + 'x' + \
                         str(temp_skip_res[i][1])  + '_output'
        # output = output_conv(output, output_ch_nums[i], scope=scope_name)
        output = output_conv(output, output_ch_nums[i], scope=scope_name)
        end_points[scope_name] = output

    output = tf.identity(output, name='decoder_output')

    return output

@slim.add_arg_scope
def GC(net, scope='GC',kw=7,kh=7,is_training=True,):

  with tf.variable_scope(scope):
    kernel_l1 = tf.Variable(tf.truncated_normal([1, kw, 128, 32], dtype=tf.float32,
                                             stddev=1e-1), name='weights_gcl1')
    # biases_l1 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                         # trainable=True, name='biases_gcl1')
    kernel_l2 = tf.Variable(tf.truncated_normal([kh, 1, 32, 8], dtype=tf.float32,
                                             stddev=1e-1), name='weights_gcl2')
    # biases_l2 = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32),
                         # trainable=True, name='biases_gcl2')

    kernel_r1 = tf.Variable(tf.truncated_normal([kh, 1, 128, 32], dtype=tf.float32,
                                             stddev=1e-1), name='weights_gcr1')
    # biases_r1 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
    #                      trainable=True, name='biases_gcr1')
    kernel_r2 = tf.Variable(tf.truncated_normal([1, kw, 32, 8], dtype=tf.float32,
                                             stddev=1e-1), name='weights_gcr2')
    # biases_r2 = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32),
    #                      trainable=True, name='biases_gcr2')

    netl = tf.nn.conv2d(net, kernel_l1, [1, 1, 1, 1], padding='SAME', name='gcl1')
    # netl = tf.nn.bias_add(netl, biases_l1)
    netl = tf.nn.conv2d(netl, kernel_l2, [1, 1, 1, 1], padding='SAME', name='gcl2')
    # netl = tf.nn.bias_add(netl, biases_l2)

    netr = tf.nn.conv2d(net, kernel_r1, [1, 1, 1, 1], padding='SAME', name='gcr1')
    # netr = tf.nn.bias_add(netr, biases_r1)
    netr = tf.nn.conv2d(netr, kernel_r2, [1, 1, 1, 1], padding='SAME', name='gcr2')
    # netr = tf.nn.bias_add(netr, biases_r2)

  return tf.identity(netl+netr,name='GC_output')

@slim.add_arg_scope
def Residual_Block(net, scope='Residual_Block',is_training=True,):
  def normal_conv(inputs, num_outputs, scope=None):
      return slim.conv2d(
                 inputs,
                 num_outputs,
                 [3, 3],
                 scope=scope,
                 weights_initializer=initializers.xavier_initializer(),
                 activation_fn=tf.nn.relu,
                 normalizer_fn=slim.batch_norm,
                 normalizer_params={'center': True, 'scale': True,
                                    'is_training': is_training},
                 biases_initializer=init_ops.zeros_initializer())
  
  with tf.variable_scope(scope):

    net_res = normal_conv(net,8)
    net_res = normal_conv(net_res,8)

  return tf.identity(net+net_res,name='GC_Residual')

@slim.add_arg_scope
def mobilenet(inputs,
              first = None,
              num_classes=21,
              prediction_fn=slim.softmax,
              reuse=None,
              scope='Mobilenet',
              base_only=False,
              use_decoder=False,
              **mobilenet_args):
  """Mobilenet model for classification, supports both V1 and V2.

  Note: default mode is inference, use mobilenet.training_scope to create
  training network.


  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    prediction_fn: a function to get predictions out of logits
      (default softmax).
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    base_only: if True will only create the base of the network (no pooling
    and no logits).
    **mobilenet_args: passed to mobilenet_base verbatim.
      - conv_defs: list of conv defs
      - multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
      - output_stride: will ensure that the last layer has at most total stride.
      If the architecture calls for more stride than that provided
      (e.g. output_stride=16, but the architecture has 5 stride=2 operators),
      it will replace output_stride with fractional convolutions using Atrous
      Convolutions.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation tensor.

  Raises:
    ValueError: Input rank is invalid.
  """
  is_training = mobilenet_args.get('is_training', False)
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Expected rank 4 input, was: %d' % len(input_shape))

  with tf.variable_scope(scope, 'Mobilenet', reuse=reuse) as scope:
    inputs = tf.identity(inputs, 'input')
    net, end_points = mobilenet_base(inputs[:,:,:,:4], scope=scope, **mobilenet_args)
    scope.reuse_variables()
    if first is None:
      net2, end_points2 = mobilenet_base(inputs[:,:,:,4:], scope=scope, **mobilenet_args)
    else:
      net2 = tf.identity(first,'first_feature')
    if base_only:
      return net, end_points

  with tf.variable_scope('MobilenetV2-Decoder', 'Mobilenet', reuse=reuse) as scope:
    net = tf.concat([net, net2], 3,name='embedding')

    net = GC(net, scope='GC',is_training=is_training)
    net = Residual_Block(net,scope='GC_Residual',is_training=is_training)

    use_decoder = FLAGS.use_decoder
    if use_decoder:
      num_skips = 3
      decoder_inputs = [end_points['layer_7'],
                        end_points['layer_4'],
                        end_points['layer_2'],
                        inputs]
      #train_size = FLAGS.train_crop_size
      skip_res = [[64, 64], [128, 128], [256, 256], [512, 512]]
      #skip_res = [[train_size[0]//16, train_size[1]//16], [train_size[0]//8, train_size[1]//8], [train_size[0]//4, train_size[1]//4], [train_size[0]//2, train_size[1]//2], [train_size[0],train_size[1]]]
      bridge_ch_nums=[8, 8, 8, 8, 4]
      output_ch_nums=[8, 8, 8, 2, 2]
      net = decoder(net, decoder_inputs, end_points,
                    num_skips=num_skips, skip_res=skip_res,
                    bridge_ch_nums=bridge_ch_nums,
                    output_ch_nums=output_ch_nums,
                    is_training=is_training,
                    scope='Decoder')
    else:      
      net = tf.identity(net, name='decoder_output')

    with tf.variable_scope('Logits'):
#      net = global_pool(net)
#      end_points['global_pool'] = net
      if not num_classes:
        return net, end_points

      # Yi Xu: remove dropout for mobilenet_sgmt
#      net = slim.dropout(net, scope='Dropout', is_training=is_training)

      # originally it is 1 x 1 x num_classes
      # since global_pool has been removed, it is now 7 x 7 x num_classes
      # Note: legacy scope name.
      logits = slim.conv2d(
          net,
          num_classes, [3, 3],
          activation_fn=None,
          normalizer_fn=None,
          biases_initializer=tf.zeros_initializer(),
          scope='Conv2d_1c_1x1')

      # since it is now 7 x 7, the squeeze here should become useless
#      logits = tf.squeeze(logits, [1, 2])
      logits = tf.image.resize_bilinear(logits, [512,512])
      
      logits = tf.identity(logits, name='output')

    end_points['Logits'] = logits

    if prediction_fn:
      end_points['Predictions'] = prediction_fn(logits, 'Predictions')

    # produce heatmap here
#    heatmap = tf.nn.softmax(logits)
#    heatmap = tf.image.resize_bilinear(
#        heatmap, FLAGS.output_size, align_corners=True)
#### the swapped version below makes border contour more smooth, but slower
#    upsampled = tf.image.resize_bilinear(
#        logits, FLAGS.output_size, align_corners=True)
#    heatmap = tf.nn.softmax(upsampled)

#  for network with dec1, upsampling is not needed here
    heatmap = tf.nn.softmax(logits)
#    heatmap.set_shape([1, 512, 512, 2])

    # deployable to coreml (slow)
#    heatmap = tf.exp(upsampled)
#    heatmap = heatmap / tf.reduce_sum(heatmap, 3, keepdims=True)

    # reduce the output channel number to one, but slice is not deployable to coreml
#    clone_batch_size = int(FLAGS.batch_size / FLAGS.num_clones)
#    heatmap = tf.slice(heatmap, [0, 0, 0, 1], [clone_batch_size, 
#                       FLAGS.output_size[0], FLAGS.output_size[1], 1])

    heatmap = tf.identity(heatmap, name='heatmap')
#    logits = tf.identity(logits, name='logits')
#    upsampled = tf.identity(upsampled, name='upsampled')

  return logits, end_points


def global_pool(input_tensor, pool_op=tf.nn.avg_pool):
  """Applies avg pool to produce 1x1 output.

  NOTE: This function is funcitonally equivalenet to reduce_mean, but it has
  baked in average pool which has better support across hardware.

  Args:
    input_tensor: input tensor
    pool_op: pooling op (avg pool is default)
  Returns:
    a tensor batch_size x 1 x 1 x depth.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size = tf.convert_to_tensor(
        [1, tf.shape(input_tensor)[1],
         tf.shape(input_tensor)[2], 1])
  else:
    kernel_size = [1, shape[1], shape[2], 1]
  output = pool_op(
      input_tensor, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
  # Recover output shape, for unknown shape.
  output.set_shape([None, 1, 1, None])
  return output


def training_scope(is_training=True,
                   weight_decay=0.00004,
                   stddev=0.09,
                   dropout_keep_prob=1.0,
                   bn_decay=0.997):
  """Defines Mobilenet training scope.

  Usage:
     with tf.contrib.slim.arg_scope(mobilenet.training_scope()):
       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

     # the network created will be trainble with dropout/batch norm
     # initialized appropriately.
  Args:
    is_training: if set to False this will ensure that all customizations are
    set to non-training mode. This might be helpful for code that is reused
    across both training/evaluation, but most of the time training_scope with
    value False is not needed.

    weight_decay: The weight decay to use for regularizing the model.
    stddev: Standard deviation for initialization, if negative uses xavier.
    dropout_keep_prob: dropout keep probability
    bn_decay: decay for the batch norm moving averages.

  Returns:
    An argument scope to use via arg_scope.
  """
  # Note: do not introduce parameters that would change the inference
  # model here (for example whether to use bias), modify conv_def instead.
  batch_norm_params = {
      'is_training': is_training,
#      'is_training': False,
#      'trainable': False,
      'decay': bn_decay,
  }

  if stddev < 0:
    weight_intitializer = slim.initializers.xavier_initializer()
  else:
    weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
      weights_initializer=weight_intitializer,
      normalizer_fn=slim.batch_norm), \
      slim.arg_scope([mobilenet_base, mobilenet], is_training=is_training),\
      slim.arg_scope([slim.batch_norm], **batch_norm_params), \
      slim.arg_scope([slim.dropout], is_training=is_training,
                     keep_prob=dropout_keep_prob), \
      slim.arg_scope([slim.conv2d], \
                     weights_regularizer=slim.l2_regularizer(weight_decay)), \
      slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as s:
    return s
