import tensorflow as tf
from tf_util_pcn import chamfer, add_train_summary, add_valid_summary
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))
import utils.tf_util3 as tf_util
import utils.pointfly as pf
from tf_ops.grouping.tf_grouping import group_point, knn_point
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
import utils.tf_util_dg as tf_util_dg
class Model:
    def __init__(self, inputs, idx, is_training):
        self.num_output_points = 16384
        labels=tf.one_hot(idx, 8)
        self.features = self.create_encoder(inputs, is_training)
        self.outputs = self.create_decoder(self.features)
        self.loss = self.create_loss(self.outputs, labels)
        self.pred=tf.argmax(self.outputs, axis=-1)

    def create_encoder(self, point_cloud, is_training):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            bn_decay=None
            """ Classification PointNet, input is BxNx3, output Bx40 """
            sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
            from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
            point_cloud = gather_point(point_cloud, farthest_point_sample(2048, point_cloud))
            batch_size = point_cloud.get_shape()[0].value
            num_point = point_cloud.get_shape()[1].value
            end_points = {}
            k = 20
            
            adj_matrix = tf_util_dg.pairwise_distance(point_cloud)
            nn_idx = tf_util_dg.knn(adj_matrix, k=k)
            edge_feature = tf_util_dg.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
            
            net = tf_util_dg.conv2d(edge_feature, 64, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='dgcnn1', bn_decay=bn_decay)
            net = tf.reduce_max(net, axis=-2, keep_dims=True)
            net1 = net
            
            adj_matrix = tf_util_dg.pairwise_distance(net)
            nn_idx = tf_util_dg.knn(adj_matrix, k=k)
            edge_feature = tf_util_dg.get_edge_feature(net, nn_idx=nn_idx, k=k)
            
            net = tf_util_dg.conv2d(edge_feature, 64, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='dgcnn2', bn_decay=bn_decay)
            net = tf.reduce_max(net, axis=-2, keep_dims=True)
            net2 = net
            
            adj_matrix = tf_util_dg.pairwise_distance(net)
            nn_idx = tf_util_dg.knn(adj_matrix, k=k)
            edge_feature = tf_util_dg.get_edge_feature(net, nn_idx=nn_idx, k=k)  
            
            net = tf_util_dg.conv2d(edge_feature, 128, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='dgcnn3', bn_decay=bn_decay)
            net = tf.reduce_max(net, axis=-2, keep_dims=True)
            net3 = net
            
            adj_matrix = tf_util_dg.pairwise_distance(net)
            nn_idx = tf_util_dg.knn(adj_matrix, k=k)
            edge_feature = tf_util_dg.get_edge_feature(net, nn_idx=nn_idx, k=k)  
            
            net = tf_util_dg.conv2d(edge_feature, 256, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='dgcnn4', bn_decay=bn_decay)
            net = tf.reduce_max(net, axis=-2, keep_dims=True)
            net4 = net
            
            net = tf_util_dg.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 512, [1, 1], 
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='agg', bn_decay=bn_decay)
            
            net = tf.concat([tf.reduce_max(net, axis=1),tf.reduce_mean(net, axis=1)],axis=-1)
            return net

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            outputs = tf_util.mlp(features, [128,8])
            outputs = tf.reshape(outputs, [-1, 8])
        return outputs

    def create_loss(self, mylabels, labels):
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=mylabels))
        
        return loss
