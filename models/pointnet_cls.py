import tensorflow as tf
from tf_util_pcn import mlp, mlp_conv, chamfer, add_train_summary, add_valid_summary

class Model:
    def __init__(self, inputs, idx, is_training):
        self.num_output_points = 16384
        labels=tf.one_hot(idx, 8)
        self.features = self.create_encoder(inputs, is_training)
        self.outputs = self.create_decoder(self.features)
        self.loss = self.create_loss(self.outputs, labels)
        self.pred=tf.argmax(self.outputs, axis=-1)

    def create_encoder(self, inputs, is_training):
        bn_decay=0.95
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
            features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = tf.reduce_max(features, axis=1, name='maxpool_1')
        return features

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            outputs = mlp(features, [128,8])
            outputs = tf.reshape(outputs, [-1, 8])
        return outputs

    def create_loss(self, mylabels, labels):
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=mylabels))
        
        return loss
