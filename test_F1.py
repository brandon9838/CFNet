import csv
import importlib
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="5"
import tensorflow as tf
import time
import sys
from io_util import read_pcd, save_pcd
from tf_util_pcn import dist_to_nearest
from visu_util import plot_pcd_nn_dist
from termcolor import colored
import models
import argparse
from utils.CFNet_utils import rotate_point_cloud,resample_pcd,axis2cor,pca_cpu,id2idx,idx2class

def test():
    inputs = tf.placeholder(tf.float32, (1, 2048, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, 16384, 3)) # dummy gt. there is no gt for kitti.
    output = tf.placeholder(tf.float32, (1, 16384, 3))
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    nearest_dist_op = dist_to_nearest(output, gt)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True
    sess = tf.Session(config=config_proto)

    outs=np.load('out_test.npy')
    gts=np.load('gt_test.npy')
    labels=np.load('label_test.npy')
    total_f1_score = 0
    f1_score_per_cat = [[],[],[],[],[],[],[],[]]

    for i in range(len(outs)):
        start = time.time()
  
        completion=outs[i,:,:].reshape(1,-1,3)
        complete=gts[i,:,:].reshape(1,-1,3)
        
        nn_dists1, nn_dists2 = sess.run(nearest_dist_op,feed_dict={output: completion, gt: complete.reshape(1,-1,3)})
        P = len(nn_dists1[nn_dists1 < 0.01]) / 16384
        R = len(nn_dists2[nn_dists2 < 0.01]) / 16384
        f1_score = 2 * P * R / (P + R + 1e-10)
        total_f1_score += f1_score

        f1_score_per_cat[labels[i]].append(f1_score)


    f1_score_per_cat=np.array(f1_score_per_cat)
    print('Average f1_score(threshold: %.4f): %.4f' % (0.01, total_f1_score / float(len(outs))))
    print('f1 score per category')
    for cat in range(8):
        print(cat,'%.4f' % (np.mean(f1_score_per_cat[cat,:])))

if __name__ == '__main__':
    test()
