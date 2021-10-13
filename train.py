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
from tf_util_pcn import chamfer
from utils.CFNet_utils import rotate_point_cloud,resample_pcd,axis2cor,pca_cpu,id2idx,id2idx_batch,idx2class

def train(args):
    ##model
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inputs_pl = tf.placeholder(tf.float32, (args.batch_size, 2048,3), 'inputs')
    inputs_cls_pl = tf.placeholder(tf.float32, [args.batch_size, 8], name='ilb')
    gt_cls_pl = tf.placeholder(tf.float32, [args.batch_size, 8], name='glb')
    gt_pl = tf.placeholder(tf.float32, (args.batch_size, 16384, 3), 'ground_truths')
    gt_pl2 = tf.placeholder(tf.float32, (args.batch_size, 256, 3), 'ground_truths2')
    gt_pl1 = tf.placeholder(tf.float32, (args.batch_size, 64 , 3), 'ground_truths1')
    target_pl = tf.placeholder(tf.float32, (args.batch_size, 3), 'attention_weight_control')
    kt_pl=tf.placeholder(tf.float32, (), 'kt')#control term proposed in BEGAN
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    G,G_c2,G_c1,avg_loss = model_module.generator(inputs_pl, target_pl,is_training_pl)
    D_real,label_real = model_module.discriminator(gt_pl,is_training_pl)
    D_gene,label_fake = model_module.discriminator(G,is_training_pl, reuse=True)
    
    ##loss
    loss_real=tf.reduce_mean(tf.squared_difference(D_real,tf.ones_like(D_real)))
    loss_fake=tf.reduce_mean(tf.squared_difference(D_gene,tf.zeros_like(D_gene)))
    loss_D_gan = loss_real + kt_pl*loss_fake
    loss_G_gan = tf.reduce_mean(tf.squared_difference(D_gene,tf.ones_like(D_gene)))
    chloss=chamfer(G,gt_pl)
    rec_loss=0.1*chamfer(G_c1,gt_pl1)+0.2*chamfer(G_c2,gt_pl2)+chloss
    loss_label_r=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt_cls_pl    , logits=label_real))
    loss_label_f=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=inputs_cls_pl, logits=label_fake))
    loss_D=loss_D_gan*0.001+loss_label_r
    loss_G=0.01*avg_loss+loss_G_gan*0.001+rec_loss+loss_label_f*0.001
    
    ##
    base_learning_rate_d=0.00005
    base_learning_rate_g=0.0001
    vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    #print(vars_D)
    vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    #print(vars_G)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_D = tf.train.AdamOptimizer(learning_rate=base_learning_rate_d).minimize(loss_D, var_list=vars_D)
        train_G = tf.train.AdamOptimizer(learning_rate=base_learning_rate_g).minimize(loss_G, var_list=vars_G)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()

    ##
    if args.restore:
        saver.restore(sess,tf.train.latest_checkpoint(args.log_dir))
        writer = tf.summary.FileWriter(args.log_dir)
        best_loss=args.prev_best
    else:
        best_loss=1.0
        sess.run(tf.global_variables_initializer())
        if os.path.exists(args.log_dir):
            delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                       % args.log_dir, 'white', 'on_red'))
            if delete_key=='y':
                os.system('rm -rf %s/*' % args.log_dir)
                os.makedirs(os.path.join(args.log_dir, 'plots'))
        else:
            os.makedirs(os.path.join(args.log_dir, 'plots'))
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(args)):
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n')                             # log of arguments
        os.system('cp models/%s.py %s' % (args.model_type, args.log_dir))                          # bkp of model def
        os.system('cp '+os.path.basename(__file__)+' %s' % args.log_dir)                           # bkp of train procedure

    ##
    with open(args.list_path) as file:
        model_list = file.read().splitlines()
    df_train, num_train = lmdb_dataflow(
        args.lmdb_train, args.batch_size, args.num_input_points, args.num_gt_points, is_training=True)
    print(num_train)
    train_gen = df_train.get_data()
    df_valid, num_valid = lmdb_dataflow(
        args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=False)
    valid_gen = df_valid.get_data()

    ##
    total_time = 0
    train_start = time.time()
    init_step = sess.run(global_step)//2
    kt = np.float32(0.)
    target_score=np.repeat(np.array([[0.5,0.25,0.25]]),args.batch_size,axis=0)
    train_loss_G=0
    train_loss_D=0
    train_loss=0
    train_counter=0
    train_acc=[]
    val_acc=[]
    test_acc=[]
    
    for step in range(init_step, args.max_step+1):
        ##data
        epoch = step * args.batch_size // num_train + 1
        ids, inputs_ori, npts, gt = next(train_gen)
        acc=[0]
        for i in npts.tolist():
            acc.append(acc[-1]+i)
        temp=[]
        gt1=[]
        gt2=[]
        for i in range(len(npts)):           
            ori=inputs_ori[0,acc[i]:acc[i+1],:]
            temp.append(resample_pcd(ori, 2048).reshape(1,2048,3))
            gt1.append(resample_pcd(gt[i,:,:], 64 ).reshape(1,64 ,3))
            gt2.append(resample_pcd(gt[i,:,:], 256).reshape(1,256,3))
        inputs_ori=np.concatenate(temp,axis=0)
        gt1=np.concatenate(gt1,axis=0)
        gt2=np.concatenate(gt2,axis=0)
        inputs, _=rotate_point_cloud(inputs_ori, gt)
        if args.shift==1:
            inputs=inputs-np.mean(inputs,axis=1).reshape(-1,1,3)
        e,v=pca_cpu(inputs)
        temp=[]
        for i in range(len(npts)):
            pcd,x,y,z=axis2cor(v[i,2,:],v[i,1,:],v[i,0,:],inputs[i,:,:])
            temp.append(pcd.reshape(1,2048,3))
        inputs=np.concatenate(temp,axis=0)
        idx=id2idx_batch(ids)
        b = np.zeros((args.batch_size, 8))
        b[np.arange(args.batch_size),idx] = 1
        idx=b

        start = time.time()
        ##discriminator        
        feed_dict = {inputs_pl: inputs, is_training_pl: True, inputs_cls_pl:idx, gt_pl: gt, gt_cls_pl:idx, gt_pl1:gt1 , gt_pl2:gt2,kt_pl:kt, target_pl:target_score}
        _, loss_dis, loss_real_val, loss_fake_val = sess.run([train_D, loss_D,loss_real,loss_fake], feed_dict=feed_dict)
        ##generator
        feed_dict = {inputs_pl: inputs, is_training_pl: True, inputs_cls_pl:idx, gt_pl: gt, gt_pl1:gt1 , gt_pl2:gt2,kt_pl:kt, target_pl:target_score}
        _, loss_gen, chamfer_loss_temp = sess.run([train_G, loss_G, chloss], feed_dict=feed_dict)
        ##control term
        kt = np.maximum(np.minimum(1., kt + 0.001 * (-0.5 * loss_real_val+loss_fake_val)), 0.0)
        
        ##
        if step % args.steps_per_visu == 0:
            samples = sess.run(G,feed_dict={inputs_pl: inputs,is_training_pl:False})
            for i in range(0, args.batch_size, args.visu_freq):
                plot_path = os.path.join(args.log_dir, 'plots', 'train_%d_step_%d.png' % (step, i))
                plot_pcd_three_views(plot_path, [inputs[i],samples[i],gt[i]], ['input','output','gt'])
        train_counter+=1
        train_loss_G+=loss_gen
        train_loss_D+=loss_dis
        train_loss+=chamfer_loss_temp
        if step % args.steps_per_print == 0:
            print('epoch %d  step %d  loss G %.5f loss D %.5f CD %.5f' % (epoch, step, loss_gen, loss_dis, chamfer_loss_temp))
                  
        if step % args.steps_per_eval == 0:
            ##validation
            train_acc.append(train_loss/train_counter)
            print('Training loss G %.5f loss D %.5f' % (train_loss_G/train_counter,train_loss_D/train_counter))
            train_loss=0
            train_counter=0
            total_loss = 0
            total_time = 0
            train_loss_G=0
            train_loss_D=0
            sess.run(tf.local_variables_initializer())
            num_eval_steps = num_valid // args.batch_size

            ##
            for eval_step in range(num_eval_steps):  
                start = time.time()
                ##
                ids, inputs_ori, npts, gt = next(valid_gen)
                idx=id2idx_batch(ids)
                acc=[0]
                for i in npts.tolist():
                    acc.append(acc[-1]+i)
                temp=[]
                for i in range(len(npts)):           
                    ori=inputs_ori[0,acc[i]:acc[i+1],:]
                    temp.append(resample_pcd(ori, 2048).reshape(1,2048,3))
                inputs_ori=np.concatenate(temp,axis=0)
                inputs, _=rotate_point_cloud(inputs_ori, gt)
                if args.shift==1:
                    inputs=inputs-np.mean(inputs,axis=1).reshape(-1,1,3)
                ##
                e,v=pca_cpu(inputs)
                temp=[]
                for i in range(len(npts)):
                    pcd,x,y,z=axis2cor(v[i,2,:],v[i,1,:],v[i,0,:],inputs[i,:,:])
                    temp.append(pcd.reshape(1,2048,3))
                inputs=np.concatenate(temp,axis=0)
                feed_dict = {inputs_pl: inputs, gt_pl: gt, is_training_pl: False}
                loss= sess.run(chloss, feed_dict=feed_dict)
                total_loss += loss
                total_time += time.time() - start
            print(colored('Validation  step %d  loss %.5f - time per batch %.4f' %
                          (step, total_loss / num_eval_steps, total_time / num_eval_steps),
                          'grey', 'on_green'))

            ##
            total_time = 0
            val_acc.append(total_loss / num_eval_steps)
            if step % args.steps_per_visu == 0:
                samples = sess.run(G,feed_dict={inputs_pl: inputs,is_training_pl:False})
                for i in range(0, args.batch_size, args.visu_freq):
                    plot_path = os.path.join(args.log_dir, 'plots','validation_%d_step_%d.png' % (step, i))
                    plot_pcd_three_views(plot_path, [inputs[i],samples[i],gt[i]], ['input','output','gt'])
            if best_loss>total_loss / num_eval_steps:
                best_loss=total_loss / num_eval_steps
                saver.save(sess, os.path.join(args.log_dir, 'model'),step)
                print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))  

            ##testing
            counter=0
            total_cd=0
            temp_out=[]
            temp_idx=[]
            for test_step, model_id in enumerate(model_list):
                idx=id2idx(model_id)
                if (test_step+1)%args.batch_size==1:
                    partial_ori = read_pcd(os.path.join(args.data_dir, 'partial', '%s.pcd' % model_id))
                    complete = read_pcd(os.path.join(args.data_dir, 'complete', '%s.pcd' % model_id)).reshape(1,-1,3)
                    partial=resample_pcd(partial_ori.reshape(-1,3), 2048).reshape(1,2048,3)
                    indexes=[idx] 
                else:
                    partial_ori =read_pcd(os.path.join(args.data_dir, 'partial', '%s.pcd' % model_id))
                    complete = np.concatenate([complete,read_pcd(os.path.join(args.data_dir, 'complete', '%s.pcd' % model_id)).reshape(1,-1,3)],axis=0)
                    partial=np.concatenate( [partial,resample_pcd(partial_ori.reshape(-1,3), 2048).reshape(1,2048,3)],axis=0 )
                    indexes.append(idx)
                if(test_step+1)%args.batch_size==0:
                    partial,_=rotate_point_cloud(partial, complete)
                    if args.shift==1:
                        partial=partial-np.mean(partial,axis=1).reshape(-1,1,3)
                    e,v=pca_cpu(partial)
                    temp=[]
                    for i in range(args.batch_size):
                        pcd,x,y,z=axis2cor(v[i,2,:],v[i,1,:],v[i,0,:],partial[i,:,:])
                        temp.append(pcd.reshape(1,2048,3))
                    partial=np.concatenate(temp,axis=0)
                    cd,my_output= sess.run([chloss,G], feed_dict={inputs_pl: partial, gt_pl: complete, is_training_pl: False})
                    counter+=1
                    total_cd += cd
            print(colored('Testing  step %d  loss %.5f' % (step, total_cd / counter), 'grey', 'on_green'))
            test_acc.append(total_cd / counter)
            a=np.array(train_acc).reshape(-1,1)
            b=np.array(val_acc).reshape(-1,1)
            c=np.array(test_acc).reshape(-1,1)
            np.savetxt(args.log_dir+"/loss.csv", np.concatenate([a,b,c],axis=-1), delimiter=",")
    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    sess.close()    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='./data/shapenet/train.lmdb')
    parser.add_argument('--lmdb_valid', default='./data/shapenet/valid.lmdb')
    parser.add_argument('--list_path', default='./data/shapenet/test.list')
    parser.add_argument('--data_dir', default='./data/shapenet/test')
    parser.add_argument('--log_dir', default='log/DIR')
    parser.add_argument('--model_type', default='CFNet')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=3000)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--prev_best', type=float, default=1.0)
    parser.add_argument('--shift', type=int, default=0)
    args = parser.parse_args()

    args.lr_decay_steps =50000*32//args.batch_size
    args.max_step       =300000*32//args.batch_size
    args.steps_per_print=100*32//args.batch_size
    args.steps_per_eval =5000*32//args.batch_size
    args.steps_per_visu =5000*32//args.batch_size
    args.steps_per_save =5000*32//args.batch_size
    args.visu_freq      =4*args.batch_size//32

    print(args)
    train(args)
