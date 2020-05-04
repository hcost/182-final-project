# impala_network.py
# Copyright (c) 2020 Daniel Grimshaw (danielgrimshaw@berkeley.edu)
#

import tensorflow as tf
from tf_agents.networks import categorical_projection_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils


class ImpalaDistributionNetwork(network.DistributionNetwork):
  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               multiplier=1):
    heads = []
    resid_blocks = []

    layer_num = 0

    def get_layer_num_str():
      nonlocal layer_num
      layer_num += 1
      return str(layer_num - 1)

    for channel in [16, 32, 32]:
      depth = channel * multiplier
      heads_block = [tf.keras.layers.Conv2D(depth, 3, padding='same', name='distribution_layer_' + get_layer_num_str()),
                     tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')]
      heads.append(heads_block)

      resid_block = [tf.keras.layers.ReLU(),
                     tf.keras.layers.Conv2D(depth, 3, padding='same', name='distribution_layer_' + get_layer_num_str()),
                     tf.keras.layers.ReLU(),
                     tf.keras.layers.Conv2D(depth, 3, padding='same', name='distribution_layer_' + get_layer_num_str()),
                     tf.keras.layers.Add()]
      resid_blocks.append(resid_block)

      resid_block = [tf.keras.layers.ReLU(),
                     tf.keras.layers.Conv2D(depth, 3, padding='same', name='distribution_layer_' + get_layer_num_str()),
                     tf.keras.layers.ReLU(),
                     tf.keras.layers.Conv2D(depth, 3, padding='same', name='distribution_layer_' + get_layer_num_str()),
                     tf.keras.layers.Add()]
      resid_blocks.append(resid_block)

    tail = [tf.keras.layers.Flatten(), tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(256, activation=tf.nn.relu, name='distribution_layer_' + get_layer_num_str())]

    projection_network = categorical_projection_network.CategoricalProjectionNetwork(output_tensor_spec)
    super().__init__(input_tensor_spec, (), projection_network.output_spec,
                                                    'impala_distribution_network')
    self.heads = heads
    self.resid_blocks = resid_blocks
    self.tail = tail
    self.projection_network = projection_network

  def call(self, inputs, step_type=None, network_state=(), training=None, mask=None):
    outer_rank = nest_utils.get_outer_rank(inputs, self.input_tensor_spec)
    batch_squash = utils.BatchSquash(outer_rank)
    out = tf.nest.map_structure(batch_squash.flatten, inputs)
    for i, top_layers in enumerate(self.heads):
      for layer in top_layers:
        out = layer(out, training=training)

      top = out
      block_1 = self.resid_blocks[2 * i]
      for layer in block_1[:-1]:
        out = layer(out, training=training)
      out = block_1[-1]([top, out])

      top = out
      block_2 = self.resid_blocks[2 * i + 1]
      for layer in block_2[:-1]:
        out = layer(out, training=training)
      out = block_2[-1]([top, out])

    for layer in self.tail:
      out = layer(out)

    out = tf.nest.map_structure(batch_squash.unflatten, out)
    out, _ = self.projection_network(out, outer_rank, training=training, mask=mask)

    return out, network_state


class ImpalaValueNetwork(network.Network):
  def __init__(self, input_tensor_spec, multiplier=1, name='impala_value_network'):
    super().__init__(input_tensor_spec, (), name)
    self.heads = []
    self.resid_blocks = []

    layer_num = 0

    def get_layer_num_str():
      nonlocal layer_num
      layer_num += 1
      return str(layer_num - 1)

    for channel in [16, 32, 32]:
      depth = channel * multiplier
      heads_block = [tf.keras.layers.Conv2D(depth, 3, padding='same', name='value_layer_' + get_layer_num_str()),
                     tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')]
      self.heads.append(heads_block)

      resid_block = [tf.keras.layers.ReLU(),
                     tf.keras.layers.Conv2D(depth, 3, padding='same', name='value_layer_' + get_layer_num_str()),
                     tf.keras.layers.ReLU(),
                     tf.keras.layers.Conv2D(depth, 3, padding='same', name='value_layer_' + get_layer_num_str()),
                     tf.keras.layers.Add()]
      self.resid_blocks.append(resid_block)

      resid_block = [tf.keras.layers.ReLU(),
                     tf.keras.layers.Conv2D(depth, 3, padding='same', name='value_layer_' + get_layer_num_str()),
                     tf.keras.layers.ReLU(),
                     tf.keras.layers.Conv2D(depth, 3, padding='same', name='value_layer_' + get_layer_num_str()),
                     tf.keras.layers.Add()]
      self.resid_blocks.append(resid_block)

    self.tail = []
    self.tail.append(tf.keras.layers.Flatten())
    self.tail.append(tf.keras.layers.ReLU())
    self.tail.append(tf.keras.layers.Dense(256, activation=tf.nn.relu, name='value_layer_' + get_layer_num_str()))
    self.tail.append(tf.keras.layers.Dense(1, activation=None, name='value_net_output'))

  def call(self, inputs, step_type=None, network_state=(), training=False):
    outer_rank = nest_utils.get_outer_rank(inputs, self.input_tensor_spec)
    batch_squash = utils.BatchSquash(outer_rank)
    out = tf.nest.map_structure(batch_squash.flatten, inputs)
    for i, top_layers in enumerate(self.heads):
      for layer in top_layers:
        out = layer(out, training=training)

      top = out
      block_1 = self.resid_blocks[2 * i]
      for layer in block_1[:-1]:
        out = layer(out, training=training)
      out = block_1[-1]([top, out])

      top = out
      block_2 = self.resid_blocks[2 * i + 1]
      for layer in block_2[:-1]:
        out = layer(out, training=training)
      out = block_2[-1]([top, out])

    for layer in self.tail[:-1]:
      out = layer(out)

    out = tf.nest.map_structure(batch_squash.unflatten, out)

    return tf.squeeze(self.tail[-1](out), -1), network_state
