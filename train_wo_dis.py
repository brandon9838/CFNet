import argparse
import datetime
import importlib
import models
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="5"
import tensorflow as tf
import time
import numpy as np
from data_util_pcn import lmdb_dataflow, get_queued_data
from termcolor import colored
from visu_util import plot_pcd_three_views
from io_util import read_pcd, save_pcd
from utils.CFNet_utils import rotate_point_cloud,resample_pcd,axis2cor,pca_cpu,id2idx,id2idx_batch,idx2class

def train(args):
    best_loss=1.0
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    target_pl = tf.placeholder(tf.float32, (args.batch_size, 3), 'target_score')
    inputs_pl = tf.placeholder(tf.float32, (args.batch_size, 2048, 3), 'inputs')

    npts_pl = tf.placeholder(tf.int32, (args.batch_size,), 'num_points')
    gt_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points, 3), 'ground_truths')
    gt_pl2 = tf.placeholder(tf.float32, (args.batch_size, 256, 3), 'ground_truths2')
    gt_pl1 = tf.placeholder(tf.float32, (args.batch_size, 64 , 3), 'ground_truths1')
    label_pl = tf.placeholder(tf.int32, (args.batch_size), 'labels')
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs_pl, gt_pl1 , gt_pl2 , gt_pl, target_pl, is_training_pl)

    with open(args.list_path) as file:
        model_list = file.read().splitlines()
    if args.lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')

    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(model.loss, global_step)

    df_train, num_train = lmdb_dataflow(
        args.lmdb_train, args.batch_size, args.num_input_points, args.num_gt_points, is_training=True)
    train_gen = df_train.get_data()
    df_valid, num_valid = lmdb_dataflow(
        args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=False)
    valid_gen = df_valid.get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()

    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
        best_loss=args.prev_best
    else:
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
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n')     
        os.system('cp models/%s.py %s' % (args.model_type, args.log_dir)) 
        os.system('cp '+os.path.basename(__file__)+' %s' % args.log_dir)   

    total_time = 0
    train_start = time.time()
    init_step = sess.run(global_step)
    target_score=np.repeat(np.array([[0.5,0.25,0.25]]),args.batch_size,axis=0)
    train_loss=0
    train_counter=0
    train_acc=[]
    val_acc=[]
    test_acc=[]

    for step in range(init_step, args.max_step+1):
        epoch = step * args.batch_size // num_train + 1
        ids, inputs_ori, npts, gt = next(train_gen)
        idx=id2idx_batch(ids)
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
        start = time.time()
        feed_dict = {inputs_pl: inputs, gt_pl1:gt1 , gt_pl2:gt2 , gt_pl: gt, target_pl:target_score, is_training_pl: True}
        _, loss = sess.run([train_op, model.loss], feed_dict=feed_dict)
        if step % args.steps_per_visu == 0:
            all_pcds = sess.run(model.visualize_ops, feed_dict=feed_dict)
            for i in range(0, args.batch_size, args.visu_freq):
                plot_path = os.path.join(args.log_dir, 'plots',
                                        'train_epoch_%d_step_%d_%s.png' % (epoch, step, ids[i]))
                pcds = [x[i] for x in all_pcds]
                plot_pcd_three_views(plot_path, pcds, model.visualize_titles)
        total_time += time.time() - start
        train_counter+=1
        train_loss+=loss
        if step % args.steps_per_print == 0:
            print('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                  (epoch, step, loss, total_time / args.steps_per_print))
            total_time = 0
        if step % args.steps_per_eval == 0:
            #validation
            train_acc.append(train_loss/train_counter)
            print(train_loss/train_counter)
            train_loss=0
            train_counter=0
            print(colored('Testing...', 'grey', 'on_green'))
            num_eval_steps = num_valid // args.batch_size
            total_loss = 0
            total_time = 0
            sess.run(tf.local_variables_initializer())
            
            for eval_step in range(num_eval_steps):
                
                start = time.time()
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
                e,v=pca_cpu(inputs)
                temp=[]
                for i in range(len(npts)):
                    pcd,x,y,z=axis2cor(v[i,2,:],v[i,1,:],v[i,0,:],inputs[i,:,:])
                    temp.append(pcd.reshape(1,2048,3))
                inputs=np.concatenate(temp,axis=0)
                feed_dict = {inputs_pl: inputs, gt_pl: gt, is_training_pl: False}
                loss= sess.run(model.chloss, feed_dict=feed_dict)
                total_loss += loss
                total_time += time.time() - start
            print(colored('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                          (epoch, step, total_loss / num_eval_steps, total_time / num_eval_steps),
                          'grey', 'on_green'))
            total_time = 0
            val_acc.append(total_loss / num_eval_steps)
            if step % args.steps_per_visu == 0:
                all_pcds = sess.run(model.visualize_ops, feed_dict=feed_dict)
                for i in range(0, args.batch_size, args.visu_freq):
                    plot_path = os.path.join(args.log_dir, 'plots',
                                            'epoch_%d_step_%d_%s.png' % (epoch, step, ids[i]))
                    pcds = [x[i] for x in all_pcds]
                    plot_pcd_three_views(plot_path, pcds, model.visualize_titles)
            if best_loss>total_loss / num_eval_steps:
                best_loss=total_loss / num_eval_steps
                saver.save(sess, os.path.join(args.log_dir, 'model'),step)
                print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))  
            ##test
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
                    start = time.time()
                    partial,_=rotate_point_cloud(partial, complete)
                    if args.shift==1:
                        partial=partial-np.mean(partial,axis=1).reshape(-1,1,3)
                    e,v=pca_cpu(partial)
                    temp=[]
                    for i in range(args.batch_size):
                        pcd,x,y,z=axis2cor(v[i,2,:],v[i,1,:],v[i,0,:],partial[i,:,:])
                        temp.append(pcd.reshape(1,2048,3))
                    partial=np.concatenate(temp,axis=0)
                    cd,my_output= sess.run([model.chloss,model.outputs], feed_dict={inputs_pl: partial, gt_pl: complete, is_training_pl: False})
                    counter+=1
                    total_cd += cd                    
            print('testing:',total_cd / counter)
            test_acc.append(total_cd / counter)
            a=np.array(train_acc).reshape(-1,1)
            b=np.array(val_acc).reshape(-1,1)
            c=np.array(test_acc).reshape(-1,1)
            np.savetxt(args.log_dir+"/loss.csv", np.concatenate([a,b,c],axis=-1), delimiter=",")
    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    sess.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='/media/bofan/linux/BOFAN/downloads/pcn-master/data/shapenet/train.lmdb')
    parser.add_argument('--lmdb_valid', default='/media/bofan/linux/BOFAN/downloads/pcn-master/data/shapenet/valid.lmdb')
    parser.add_argument('--list_path', default='/media/bofan/linux/BOFAN/downloads/pcn-master/data/shapenet/test.list')
    parser.add_argument('--data_dir', default='/media/bofan/linux/BOFAN/downloads/pcn-master/data/shapenet/test')
    parser.add_argument('--log_dir', default='log/DIR')
    parser.add_argument('--model_type', default='CFNet_wo_dis')
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
