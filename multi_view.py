import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
import numpy as np
from data_util_pcn import lmdb_dataflow, get_queued_data
from termcolor import colored
from visu_util import plot_pcd_three_views
from io_util import read_pcd, save_pcd
import matplotlib.pyplot as plt
from tf_util_pcn import chamfer_separ, chamfer,dist_to_nearest
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
import utils.pointfly as pf
from utils.CFNet_utils import rotate_point_cloud,resample_pcd,axis2cor,pca_cpu,id2idx_batch

def get_density(layer_pts):
    qrs = gather_point(layer_pts, farthest_point_sample(256, layer_pts))
    indices = pf.knn_indices_general(qrs, layer_pts, int(16), True)
    nn_pts = tf.gather_nd(layer_pts, indices)
    nn_pts = nn_pts-tf.expand_dims(qrs,axis=-2)
    norm=tf.norm(nn_pts,axis=-1)
    my_dist=tf.reduce_mean(norm,axis=-1)
    my_dist=tf.reduce_mean(my_dist,axis=-1)
    return my_dist

def test(args):

    model_graph = tf.Graph()
    sess = tf.Session(graph=model_graph)
    pointnet_graph = tf.Graph()
    pointnet_sess = tf.Session(graph=pointnet_graph)
    dgcnn_graph = tf.Graph()
    dgcnn_sess = tf.Session(graph=dgcnn_graph)
    bs=8
    with sess.as_default():
        with model_graph.as_default():
            is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
            inputs_pl = tf.placeholder(tf.float32, (bs, 2048,3), 'inputs')
            gt_pl = tf.placeholder(tf.float32, (bs, 16384, 3), 'ground_truths')
            out_pl = tf.placeholder(tf.float32, (bs, 16384, 3), 'out')
            fea_pl= tf.placeholder(tf.float32, (bs, 1024), 'fea')
            target_pl=tf.placeholder(tf.float32, (1,3), 'target')
            model_module = importlib.import_module('.%s' % args.model_type, 'models')
            G,G_c2,G_c1, fea_out = model_module.generator(inputs_pl, target_pl, fea_pl,is_training_pl)
            D_real,gt_real = model_module.discriminator(gt_pl,is_training_pl)

            D_gene,gt_fake = model_module.discriminator(G,is_training_pl, reuse=True)
            
            dist=get_density(inputs_pl)
            
            cd_op=chamfer_separ(G,gt_pl)
            
            
            nearest_dist_op = dist_to_nearest(G,gt_pl)
            
            saver = tf.train.Saver()
            saver.restore(sess, args.checkpoint)
    with pointnet_sess.as_default():
        with pointnet_graph.as_default():
            gt_ptn = tf.placeholder(tf.float32, (bs, args.num_gt_points, 3), 'ground_truth_ptn')
            label_ptn = tf.placeholder(tf.int32, (bs), 'labels_ptn')
            is_training_ptn = tf.placeholder(tf.bool, shape=(), name='is_training_ptn')
            model_module_ptn = importlib.import_module('.%s' % args.model_type_ptn, 'models')
            model_ptn = model_module_ptn.Model(gt_ptn, label_ptn, is_training_ptn)
            saver_ptn = tf.train.Saver()
            saver_ptn.restore(pointnet_sess, args.checkpoint_ptn)
    with dgcnn_sess.as_default():
        with dgcnn_graph.as_default():
            gt_dg = tf.placeholder(tf.float32, (bs, args.num_gt_points, 3), 'ground_truth_dg')
            label_dg = tf.placeholder(tf.int32, (bs), 'labels_dg')
            is_training_dg = tf.placeholder(tf.bool, shape=(), name='is_training_dg')
            model_module_dg = importlib.import_module('.%s' % args.model_type_dg, 'models')
            model_dg = model_module_dg.Model(gt_dg, label_dg, is_training_dg)
            saver_dg = tf.train.Saver()
            saver_dg.restore(dgcnn_sess, args.checkpoint_dg)
            
    #Baseline, Most Points, Low density, AVG     
    cd_temp=[[],[],[],[]]
    F_temp=[[],[],[],[]]
    ptn_temp=[[],[],[],[]]
    dg_temp=[[],[],[],[]]
    idx_temp=[]
    df_train, num_train = lmdb_dataflow(
        args.lmdb_train, 8, 3000, 16384, is_training=False)
    train_gen = df_train.get_data()
    
    
    
    #for step in range(500):
    for step in range(num_train//8):
        
        ids, inputs_ori, npts, gt = next(train_gen)
        idx=id2idx_batch(ids)
        
        
        ## process
        acc=[0]
        for i in npts.tolist():
            acc.append(acc[-1]+i)
        temp=[]
        gt1=[]
        gt2=[]
        for i in range(len(npts)):           
            ori=inputs_ori[0,acc[i]:acc[i+1],:]
            temp.append(resample_pcd(ori, 2048).reshape(1,2048,3))
            
        inputs_ori=np.concatenate(temp,axis=0)
        
        inputs, _=rotate_point_cloud(inputs_ori, gt)
        e,v=pca_cpu(inputs)
        temp=[]
        for i in range(len(npts)):
            pcd,x,y,z=axis2cor(v[i,2,:],v[i,1,:],v[i,0,:],inputs[i,:,:])
            temp.append(pcd.reshape(1,2048,3))
        inputs=np.concatenate(temp,axis=0)
        ## process
        
        ##compute
        fea_out_val,dist_val=sess.run([fea_out,dist],feed_dict={inputs_pl:inputs,is_training_pl:False})
        cd_ori_val,[nn_dists_ori1,nn_dists_ori2],out_ori_val=sess.run([cd_op,nearest_dist_op,G],feed_dict={gt_pl:gt,fea_pl:fea_out_val,is_training_pl:False})
        fea_avg=np.repeat(np.mean(fea_out_val,axis=0,keepdims=True),bs,axis=0)
        cd_avg_val,[nn_dists_avg1,nn_dists_avg2],out_avg_val=sess.run([cd_op,nearest_dist_op,G],feed_dict={gt_pl:gt,fea_pl:fea_avg,is_training_pl:False})
        
                        
        P = np.sum(nn_dists_ori1 < 0.01,axis=1) / 16384.0
        R = np.sum(nn_dists_ori2 < 0.01,axis=1) / 16384.0
        f1_score_ori = 2 * P * R / (P + R)
        
        P = np.sum(nn_dists_avg1 < 0.01,axis=1) / 16384.0
        R = np.sum(nn_dists_avg2 < 0.01,axis=1) / 16384.0
        f1_score_avg = 2 * P * R / (P + R)
        
        point_pred_ori_val=pointnet_sess.run(model_ptn.pred,feed_dict={gt_ptn:out_ori_val,is_training_ptn:False})
        point_pred_avg_val=pointnet_sess.run(model_ptn.pred,feed_dict={gt_ptn:out_avg_val,is_training_ptn:False})
        dgcnn_pred_ori_val=dgcnn_sess.run(model_dg.pred,feed_dict={gt_dg:out_ori_val,is_training_dg:False})
        dgcnn_pred_avg_val=dgcnn_sess.run(model_dg.pred,feed_dict={gt_dg:out_avg_val,is_training_dg:False})
        
        ptn_acc_ori=np.sum(idx==point_pred_ori_val)
        ptn_acc_avg=np.sum(idx==point_pred_avg_val)
        dg_acc_ori =np.sum(idx==dgcnn_pred_ori_val)
        dg_acc_avg =np.sum(idx==dgcnn_pred_avg_val)
        
        ## record
        idx_temp.append(idx.reshape(-1))
        cd_temp[0].append(cd_ori_val)
        
        F_temp[0].append(f1_score_ori)
        ptn_temp[0].append(point_pred_ori_val)
        dg_temp[0].append(dgcnn_pred_ori_val)
        
        choice=np.argmax(npts)
        cd_temp[1].append(np.repeat(cd_ori_val[choice],bs,axis=0))
        F_temp[1].append(np.repeat(f1_score_ori[choice],bs,axis=0))
        ptn_temp[1].append(np.repeat(point_pred_ori_val[choice],bs,axis=0))
        dg_temp[1].append(np.repeat(dgcnn_pred_ori_val[choice],bs,axis=0))
        
        choice=np.argmax(dist_val)
        cd_temp[2].append(np.repeat(cd_ori_val[choice],bs,axis=0))
        F_temp[2].append(np.repeat(f1_score_ori[choice],bs,axis=0))
        ptn_temp[2].append(np.repeat(point_pred_ori_val[choice],bs,axis=0))
        dg_temp[2].append(np.repeat(dgcnn_pred_ori_val[choice],bs,axis=0))
        
        cd_temp[3].append(cd_avg_val)
        F_temp[3].append(f1_score_avg)
        ptn_temp[3].append(point_pred_avg_val)
        dg_temp[3].append(dgcnn_pred_avg_val)
        
        if (step+1)%1000==0:
            print(step)
            np.save('idx_AGG',np.concatenate(idx_temp,axis=0))
            np.save('cd_AGG',np.array(cd_temp).reshape(4,-1))
            np.save('F_AGG',np.array(F_temp).reshape(4,-1))
            np.save('ptn_AGG',np.array(ptn_temp).reshape(4,-1))
            np.save('dgcnn_AGG',np.array(dg_temp).reshape(4,-1))
            
    
    
    np.save('idx_AGG',np.concatenate(idx_temp,axis=0))
    np.save('cd_AGG',np.array(cd_temp).reshape(4,-1))
    np.save('F_AGG',np.array(F_temp).reshape(4,-1))
    np.save('ptn_AGG',np.array(ptn_temp).reshape(4,-1))
    np.save('dg_AGG',np.array(dg_temp).reshape(4,-1))
    sess.close()
    pointnet_sess.close()
    dgcnn_sess.close()    

    a=np.load('idx_AGG.npy').reshape(1,-1)
    b=np.load('cd_AGG.npy')
    c=np.load('F_AGG.npy')
    d=np.load('ptn_AGG.npy')
    e=np.load('dg_AGG.npy')
    print(a.shape,b.shape,c.shape,d.shape,e.shape)
    print('finish')
    
    
    cls_cd_baseline          =[[],[],[],[],[],[],[],[]]
    cls_F_baseline           =[[],[],[],[],[],[],[],[]]
    cls_pointnet_acc_baseline=[[],[],[],[],[],[],[],[]]
    cls_dgcnn_acc_baseline   =[[],[],[],[],[],[],[],[]]
    cls_cd_max               =[[],[],[],[],[],[],[],[]]
    cls_F_max                =[[],[],[],[],[],[],[],[]]
    cls_pointnet_acc_max     =[[],[],[],[],[],[],[],[]]
    cls_dgcnn_acc_max        =[[],[],[],[],[],[],[],[]]
    cls_cd_avg               =[[],[],[],[],[],[],[],[]]
    cls_F_avg                =[[],[],[],[],[],[],[],[]]
    cls_pointnet_acc_avg     =[[],[],[],[],[],[],[],[]]
    cls_dgcnn_acc_avg        =[[],[],[],[],[],[],[],[]]
    d=(d==a)
    e=(e==a)
    for i in range(a.shape[1]):
        idx=int(a[0,i])
        cls_cd_baseline[idx].append(b[0,i])    
        cls_cd_max[idx].append(b[1,i])
        cls_cd_avg[idx].append(b[3,i])
        cls_F_baseline[idx].append(c[0,i])    
        cls_F_max[idx].append(c[1,i])
        cls_F_avg[idx].append(c[3,i])
        cls_pointnet_acc_baseline[idx].append(d[0,i])    
        cls_pointnet_acc_max[idx].append(d[1,i])
        cls_pointnet_acc_avg[idx].append(d[3,i])
        cls_dgcnn_acc_baseline[idx].append(e[0,i])    
        cls_dgcnn_acc_max[idx].append(e[1,i])
        cls_dgcnn_acc_avg[idx].append(e[3,i])
    print('----------Baseline Model:----------')
    cd=[]
    F=[]
    pointnet_acc=[]
    dgcnn_acc=[]
    for i in range(8):
        cd.append(sum(cls_cd_baseline[i])/len(cls_cd_baseline[i]))
        F.append(sum(cls_F_baseline[i])/len(cls_F_baseline[i]))
        pointnet_acc.append(sum(cls_pointnet_acc_baseline[i])/len(cls_pointnet_acc_baseline[i]))
        dgcnn_acc.append(sum(cls_dgcnn_acc_baseline[i])/len(cls_dgcnn_acc_baseline[i]))
    print('CD           :',sum(cd)/len(cd))
    print('F-score      :',sum(F)/len(F))
    print('Acc. PointNet:',sum(pointnet_acc)/len(pointnet_acc))
    print('Acc. DGCNN   :',sum(dgcnn_acc)/len(dgcnn_acc))
    
    print('---------Most Points:----------')
    cd=[]
    F=[]
    pointnet_acc=[]
    dgcnn_acc=[]
    for i in range(8):
        cd.append(sum(cls_cd_max[i])/len(cls_cd_max[i]))
        F.append(sum(cls_F_max[i])/len(cls_F_max[i]))
        pointnet_acc.append(sum(cls_pointnet_acc_max[i])/len(cls_pointnet_acc_max[i]))
        dgcnn_acc.append(sum(cls_dgcnn_acc_max[i])/len(cls_dgcnn_acc_max[i]))
    print('CD           :',sum(cd)/len(cd))
    print('F-score      :',sum(F)/len(F))
    print('Acc. PointNet:',sum(pointnet_acc)/len(pointnet_acc))
    print('Acc. DGCNN   :',sum(dgcnn_acc)/len(dgcnn_acc))

    print('----------Feature Average:----------')
    cd=[]
    F=[]
    pointnet_acc=[]
    dgcnn_acc=[]
    for i in range(8):
        cd.append(sum(cls_cd_avg[i])/len(cls_cd_avg[i]))
        F.append(sum(cls_F_avg[i])/len(cls_F_avg[i]))
        pointnet_acc.append(sum(cls_pointnet_acc_avg[i])/len(cls_pointnet_acc_avg[i]))
        dgcnn_acc.append(sum(cls_dgcnn_acc_avg[i])/len(cls_dgcnn_acc_avg[i]))
    print('CD           :',sum(cd)/len(cd))
    print('F-score      :',sum(F)/len(F))
    print('Acc. PointNet:',sum(pointnet_acc)/len(pointnet_acc))
    print('Acc. DGCNN   :',sum(dgcnn_acc)/len(dgcnn_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='./data/shapenet/train.lmdb')
    parser.add_argument('--model_type', default='CFNet_multiview')
    parser.add_argument('--checkpoint', default='log/CFNet_rot/model')
    parser.add_argument('--model_type_ptn', default='pointnet_cls')
    parser.add_argument('--checkpoint_ptn', default='log/pointnet_cls/model')
    parser.add_argument('--model_type_dg', default='dgcnn_cls')
    parser.add_argument('--checkpoint_dg', default='log/dgcnn_cls/model')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    args = parser.parse_args()

    test(args)
