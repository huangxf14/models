"""
Implementation of Mobilenet V2 for semantic segmentation.
Adjusted form slim/nets/mobilenet/mobilenet_v2.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

from nets.mobilenet import conv_blocks as ops

from net import mobilenet_sgmt as lib

slim = tf.contrib.slim
op = lib.op

expand_input = ops.expand_input_by_factor

# pyformat: disable
# Architecture: https://arxiv.org/abs/1801.04381
V2_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
#        (slim.batch_norm,): {'center': True, 'scale': False},
#        (slim.batch_norm,): {'center': True, 'scale': True, 'fused': False},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 
#            'normalizer_fn': None,
            'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
#            'normalizer_fn': None,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=16, kernel_size=[3, 3]), # conv, stride=2
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=8),                                # conv, 1/2 skip input, layer_2
        op(ops.expanded_conv, stride=2, num_outputs=8),   # conv_1, stride=2
        op(ops.expanded_conv, stride=1, num_outputs=8),   # conv_2, 1/4 skip input, layer_4
        op(ops.expanded_conv, stride=2, num_outputs=16),  # conv_3, stride=2
        op(ops.expanded_conv, stride=1, num_outputs=16),  # conv_4
        op(ops.expanded_conv, stride=1, num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=24),  # conv_6, stride=2
        op(ops.expanded_conv, stride=1, num_outputs=24),  # conv_7
        op(ops.expanded_conv, stride=1, num_outputs=24),
        op(ops.expanded_conv, stride=1, num_outputs=24),
        op(ops.expanded_conv, stride=1, num_outputs=32),  # conv_10
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=56),  # conv_13, stride=2
        op(ops.expanded_conv, stride=1, num_outputs=56),
        op(ops.expanded_conv, stride=1, num_outputs=56),
        op(ops.expanded_conv, stride=1, num_outputs=112), # conv_16: 112
        op(slim.conv2d, stride=1, kernel_size=[3, 3], num_outputs=64), 
    ],
)
# pyformat: enable


@slim.add_arg_scope
def mobilenet(input_tensor,
              first = None,
              num_classes=21,
              depth_multiplier=1.0,
              scope='MobilenetV2',
              conv_defs=None,
              finegrain_classification_mode=False,
              min_depth=None,
              divisible_by=None,
              **kwargs):
  """Creates mobilenet V2 network.

  Inference mode is created by default. To create training use training_scope
  below.

  with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
     logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

  Args:
    input_tensor: The input tensor
    num_classes: number of classes
    depth_multiplier: The multiplier applied to scale number of
    channels in each layer. Note: this is called depth multiplier in the
    paper but the name is kept for consistency with slim's model builder.
    scope: Scope of the operator
    conv_defs: Allows to override default conv def.
    finegrain_classification_mode: When set to True, the model
    will keep the last layer large even for small multipliers. Following
    https://arxiv.org/abs/1801.04381
    suggests that it improves performance for ImageNet-type of problems.
      *Note* ignored if final_endpoint makes the builder exit earlier.
    min_depth: If provided, will ensure that all layers will have that
    many channels after application of depth multiplier.
    divisible_by: If provided will ensure that all layers # channels
    will be divisible by this number.
    **kwargs: passed directly to mobilenet.mobilenet:
      prediction_fn- what prediction function to use.
      reuse-: whether to reuse variables (if reuse set to true, scope
      must be given).
  Returns:
    logits/endpoints pair

  Raises:
    ValueError: On invalid arguments
  """
  if conv_defs is None:
    conv_defs = V2_DEF
  if 'multiplier' in kwargs:
    raise ValueError('mobilenetv2 doesn\'t support generic '
                     'multiplier parameter use "depth_multiplier" instead.')
  if finegrain_classification_mode:
    conv_defs = copy.deepcopy(conv_defs)
    if depth_multiplier < 1:
      conv_defs['spec'][-1].params['num_outputs'] /= depth_multiplier

  depth_args = {}
  # NB: do not set depth_args unless they are provided to avoid overriding
  # whatever default depth_multiplier might have thanks to arg_scope.
  if min_depth is not None:
    depth_args['min_depth'] = min_depth
  if divisible_by is not None:
    depth_args['divisible_by'] = divisible_by

  with slim.arg_scope((lib.depth_multiplier,), **depth_args):
    return lib.mobilenet(
        input_tensor,
        first,
        num_classes=num_classes,
        conv_defs=conv_defs,
        scope=scope,
        multiplier=depth_multiplier,
        **kwargs)


@slim.add_arg_scope
def mobilenet_base(input_tensor, depth_multiplier=1.0, **kwargs):
  """Creates base of the mobilenet (no pooling and no logits) ."""
  return mobilenet(input_tensor,
                   depth_multiplier=depth_multiplier,
                   base_only=True, **kwargs)


def training_scope(**kwargs):
  """Defines MobilenetV2 training scope.

  Usage:
     with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

  with slim.

  Args:
    **kwargs: Passed to mobilenet.training_scope. The following parameters
    are supported:
      weight_decay- The weight decay to use for regularizing the model.
      stddev-  Standard deviation for initialization, if negative uses xavier.
      dropout_keep_prob- dropout keep probability
      bn_decay- decay for the batch norm moving averages.

  Returns:
    An `arg_scope` to use for the mobilenet v2 model.
  """
  return lib.training_scope(**kwargs)


__all__ = ['training_scope', 'mobilenet_base', 'mobilenet', 'V2_DEF']
