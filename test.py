import argparse
import csv
import importlib
import models
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="5"
import tensorflow as tf
from io_util import read_pcd, save_pcd
from tf_util_pcn import chamfer, earth_mover
from visu_util import plot_pcd_three_views
from utils.CFNet_utils import rotate_point_cloud,resample_pcd,axis2cor,pca_cpu,id2idx,idx2class


    
def test(args):
    is_training_pl    = tf.placeholder(tf.bool, shape=(), name='is_training')
    target_pl         = tf.placeholder(tf.float32, (1, 3), 'target_score')
    inputs_pl         = tf.placeholder(tf.float32, (1, 2048,3), 'inputs')
    gt_pl             = tf.placeholder(tf.float32, (1, 16384, 3), 'ground_truths')
    model_module      = importlib.import_module('.%s' % args.model_type, 'models')
    G,G1,G2,avg_loss  = model_module.generator(inputs_pl,target_pl,is_training_pl)
    D_real,label_real = model_module.discriminator(gt_pl,is_training_pl)
    D_gene,label_fake = model_module.discriminator(G,is_training_pl, reuse=True)
    cd_op             = chamfer(G, gt_pl)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)
    os.makedirs(args.results_dir, exist_ok=True)
    
    with open(args.list_path) as file:
        model_list = file.read().splitlines()

    
    total_cd   = 0
    cd_per_cat = {}    
    temp_out   = []
    temp_gt    = []
    temp_idx   = []
    for i, model_id in enumerate(model_list):

        idx=id2idx(model_id)
        partial_ori = read_pcd(os.path.join(args.data_dir, 'partial', '%s.pcd' % model_id))
        complete = read_pcd(os.path.join(args.data_dir, 'complete', '%s.pcd' % model_id))
        partial_ori=resample_pcd(partial_ori.reshape(-1,3), 2048).reshape(1,2048,3)
        gt1=resample_pcd(complete.reshape(-1,3), 64).reshape(1,64,3)
        gt2=resample_pcd(complete.reshape(-1,3), 256).reshape(1,256,3)
        partial, _=rotate_point_cloud(partial_ori, complete)
        if args.shift==1:
            partial=partial-np.mean(partial,axis=1).reshape(1,1,3)
        partial_input=partial

        e,v=pca_cpu(partial)
        pcd,x,y,z=axis2cor(v[0,2,:],v[0,1,:],v[0,0,:],partial[0,:,:])
        partial=pcd.reshape(1,2048,3)
        completion,cd= sess.run([G,cd_op], feed_dict={inputs_pl: partial, gt_pl:complete.reshape(1,-1,3), is_training_pl: False})

        temp_gt.append(complete)
        temp_out.append(completion)
        temp_idx.append(idx)
        total_cd += cd

        
        if not cd_per_cat.get(idx):
            cd_per_cat[idx] = []
        cd_per_cat[idx].append(cd)
        
        if i % args.plot_freq == 0:
            print('print:',idx2class(idx))
            os.makedirs(os.path.join(args.results_dir, 'plots', idx2class(idx)), exist_ok=True)
            plot_path = os.path.join(args.results_dir, 'plots', idx2class(idx), '%s.png' % str(i))
            plot_pcd_three_views(plot_path, [partial_input[0], completion[0], complete],
                                 ['input', 'output', 'ground truth'],
                                 'CD %.5f' % (cd),
                                 [5, 0.5, 0.5])
    sess.close()

    
    np.save('label_test', np.array(temp_idx))
    np.save('out_test', np.array(temp_out))
    np.save('gt_test',np.array(temp_gt))

    print('Average Chamfer distance: %.5f' % (total_cd / len(model_list)))
    print('Chamfer distance per category')
    for idx in cd_per_cat.keys():
        print(idx2class(idx), '%.5f' % np.mean(cd_per_cat[idx]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path', default='/media/bofan/linux/BOFAN/downloads/pcn-master/data/shapenet/test.list')
    parser.add_argument('--data_dir', default='/media/bofan/linux/BOFAN/downloads/pcn-master/data/shapenet/test')
    parser.add_argument('--model_type', default='CFNet')
    parser.add_argument('--checkpoint', default='log/CFNet_rot/model')
    parser.add_argument('--results_dir', default='results/CFNet_rot')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--plot_freq', type=int, default=50)
    parser.add_argument('--shift', type=int, default=0)
    args = parser.parse_args()

    test(args)

