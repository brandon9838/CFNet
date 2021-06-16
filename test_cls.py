import argparse
import csv
import importlib
import models
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="5"
import tensorflow as tf
import time
from io_util import read_pcd, save_pcd
from tf_util_pcn import chamfer, earth_mover
from visu_util import plot_pcd_three_views
from utils.CFNet_utils import rotate_point_cloud,resample_pcd,axis2cor,pca_cpu,id2idx,idx2class


def test(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    gt = tf.placeholder(tf.float32, (1, 16384, 3), 'ground_truths')
    label_pl = tf.placeholder(tf.int32, (1), 'labels')
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(gt, label_pl, is_training_pl)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    num_per_cat = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    right_per_cat = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    pcds=np.load('out_test.npy')
    idxs=np.load('label_test.npy')

    for i in range(len(pcds)):
        ans= sess.run(model.pred, feed_dict={gt:pcds[i,:,:].reshape(1,16384,3), label_pl:[idxs[i]], is_training_pl: False})
        num_per_cat[idxs[i]]+=1
        if ans==idxs[i]:
            right_per_cat[int(ans)]+=1
    sess.close()
    
    for i in range(len(num_per_cat)):
        print(idx2class(i), ' : %.4f' % (right_per_cat[i]/num_per_cat[i]))
    print('Total: %.4f' % (sum(right_per_cat)/sum(num_per_cat)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='pointnet_cls')
    parser.add_argument('--checkpoint', default='log/pointnet_cls/model')
    args = parser.parse_args()
    test(args)
