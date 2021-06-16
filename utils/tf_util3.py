import numpy as np
import tensorflow as tf
import transforms3d
import random
import math
def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed
def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
def mlp_conv(inputs, layer_dims, bn=None, bn_params=None,name=''):
    with tf.variable_scope(name):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf.contrib.layers.conv1d(
                inputs, num_out_channel,
                kernel_size=1,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='conv_%d' % i)
        outputs = tf.contrib.layers.conv1d(
            inputs, layer_dims[-1],
            kernel_size=1,
            activation_fn=None,
            scope='conv_%d' % (len(layer_dims) - 1))
        return outputs
def mlp(features, layer_dims, bn=None, bn_params=None,name=''):
    with tf.variable_scope(name):
        for i, num_outputs in enumerate(layer_dims[:-1]):
            features = tf.layers.dense(
                features, num_outputs,
                activation=tf.nn.relu,
                name=name+str(i))
        outputs = tf.layers.dense(
                features, layer_dims[-1],
                activation=None,
                name=name+str(len(layer_dims) - 1))
        
        return outputs
'''
def conv2d(inputs,
           num_output_channels,
           kernel_size,
           name=None,
           stride=[1, 1],
           padding='SAME',
           weight_decay=0.00001,
           activation_fn=tf.nn.relu):
    with tf.variable_scope(name) as sc:
        
        outputs = tf.layers.conv2d(
                    inputs,
                    num_output_channels, 
                    kernel_size, 
                    strides=stride, 
                    padding=padding, 
                    name=name)
        if activation_fn is not None:
            outputs=activation_fn(outputs)
        return outputs
'''
def conv2d(inputs,
           num_output_channels,
           kernel_size,
           name,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_multi_GPU=False,
           gn=False,
           G=32):
  
  with tf.variable_scope(name) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        if is_multi_GPU:
          outputs = batch_norm_template_multiGPU(outputs, is_training,
                                                 'bn', [0,1,2], bn_decay)
        else:
          outputs = batch_norm_template(outputs, is_training,'bn', [0,1,2], bn_decay)
      if gn:
        outputs = group_norm_for_conv(outputs, G=G, scope='gn')
      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

