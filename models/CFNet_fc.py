import tensorflow as tf
from tf_util_pcn import mlp, mlp_conv, chamfer, add_train_summary, add_valid_summary
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))
import utils.tf_util as tf_util
import utils.pointfly as pf
from tf_ops.grouping.tf_grouping import group_point, knn_point
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point



def sub_module(pts, fts_prev_RI, fts_prev_AE, rot_prev_AE, qrs, qrs_fea_AE, qrs_rot_AE, is_training, tag, K, D, P, C, use_fea_nn):
    bn_decay=None
    with tf.variable_scope(tag+'knn', reuse=tf.AUTO_REUSE):
        indices = pf.knn_indices_general(qrs, pts, int(K), True)
        nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')
        
        nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')
        nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local') 
        dists_local = tf.norm(nn_pts_local, axis=-1, keepdims=True)  # dist to center

        mean_local = tf.reduce_mean(nn_pts, axis=-2, keepdims=True)
        nn_pts_local_mean = tf.subtract(nn_pts, mean_local, name=tag + 'nn_pts_local_mean') 
        dists_local_mean = tf.norm(nn_pts_local_mean, axis=-1, keepdims=True) # dist to local mean
        
    with tf.variable_scope(tag+'MLP_fea', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(tag+'MLP_fea1', reuse=tf.AUTO_REUSE):
            fea_MLP     = mlp_conv(qrs,[C//4,C//2])
            fea_MLP_glb = tf.reduce_max(fea_MLP, axis=1, keep_dims=True, name='maxpool_0')
            fea_MLP     = tf.concat([fea_MLP, tf.tile(fea_MLP_glb, [1, tf.shape(qrs)[1], 1])], axis=2)
        with tf.variable_scope(tag+'MLP_fea2', reuse=tf.AUTO_REUSE):    
            fea_MLP=mlp_conv(fea_MLP,[C,C*2])
        
    with tf.variable_scope(tag+'RI_fts', reuse=tf.AUTO_REUSE):
        vec = mean_local - nn_pts_center
        vec_dist = tf.norm(vec, axis=-1, keepdims =True)
        vec_norm = tf.divide(vec, vec_dist)
        vec_norm = tf.where(tf.is_nan(vec_norm), tf.ones_like(vec_norm) * 0, vec_norm) 

        nn_pts_local_proj = tf.matmul(nn_pts_local, vec_norm, transpose_b=True)
        nn_pts_local_proj_dot = tf.divide(nn_pts_local_proj, dists_local)
        nn_pts_local_proj_dot = tf.where(tf.is_nan(nn_pts_local_proj_dot), tf.ones_like(nn_pts_local_proj_dot) * 0, nn_pts_local_proj_dot)  # check nan

        nn_pts_local_proj_2 = tf.matmul(nn_pts_local_mean, vec_norm, transpose_b=True)
        nn_pts_local_proj_dot_2 = tf.divide(nn_pts_local_proj_2, dists_local_mean)
        nn_pts_local_proj_dot_2 = tf.where(tf.is_nan(nn_pts_local_proj_dot_2), tf.ones_like(nn_pts_local_proj_dot_2) * 0, nn_pts_local_proj_dot_2)  # check nan
    
        vec = tf.squeeze(mean_local,axis=-2) 
        vec_norm=tf.expand_dims(tf.norm(vec,axis=-1),axis=-1)
        my_fea=tf.tile(tf.expand_dims(vec_norm,axis=-1), [1, 1, K, 1])
        nn_fts_RI = tf.concat([dists_local, dists_local_mean, nn_pts_local_proj_dot, nn_pts_local_proj_dot_2,my_fea], axis=-1) # d0 d1 a0 a1
    if use_fea_nn or fts_prev_AE is not None:
        with tf.variable_scope(tag+'AE_fts_fea', reuse=tf.AUTO_REUSE):
            rot_mtx_AE= tf.reshape(rot_prev_AE,shape=(rot_prev_AE.shape[0],rot_prev_AE.shape[1],-1))
            if use_fea_nn: 
                indices_AE= pf.knn_indices_general(qrs_fea_AE, fts_prev_AE, int(K), True, tmp_k=1)
            else:
                indices_AE=indices
            nn_pts_AE = tf.gather_nd(pts, indices_AE, name=tag + 'nn_pts_AE')
            
            nn_rot_AE = tf.gather_nd(rot_mtx_AE, indices_AE, name=tag + 'nn_rot_AE')
            nn_rot_AE = tf.reshape(nn_rot_AE,shape=(nn_rot_AE.shape[0],nn_rot_AE.shape[1],nn_rot_AE.shape[2],3,3))
            nn_rot_AE = tf.transpose(nn_rot_AE,perm=[0,1,2,4,3])
            
            rot_mtx_temp_AE = tf.expand_dims(tf.reshape(qrs_rot_AE,[qrs_rot_AE.shape[0],qrs_rot_AE.shape[1],3,3]),axis=-3)
            rot_mtx_temp_AE = tf.tile(rot_mtx_temp_AE,[1,1,K,1,1])
            trans_AE = tf.expand_dims(qrs,axis=-2)-nn_pts_AE
            fts_prev_AE = tf.gather_nd(fts_prev_AE, indices_AE, name=tag + 'nn_fts_AE')
            
            #print(rot_mtx_temp.shape)
            #print(nn_rot_mtx.shape)
            
            rotate_AE = tf.matmul(rot_mtx_temp_AE,nn_rot_AE)
            trans_AE  = tf.squeeze(tf.matmul(tf.expand_dims(trans_AE,axis=-2),tf.transpose(rot_mtx_temp_AE,perm=[0,1,2,4,3])),axis=-2)
            rotate_AE = tf.reshape(rotate_AE,shape=(rotate_AE.shape[0],rotate_AE.shape[1],rotate_AE.shape[2],-1))
    if use_fea_nn:
        nn_fts_AE = tf.concat([rotate_AE,trans_AE,fts_prev_AE,my_fea],axis=-1)
    else:
        with tf.variable_scope(tag+'AE_fts_pos', reuse=tf.AUTO_REUSE):
            #get normalized om
            indices_AE=indices
            vec_dist = tf.norm(qrs, axis=-1, keepdims =True)
            vec_norm = tf.divide(qrs, vec_dist)
            op_norm = tf.where(tf.is_nan(vec_norm), tf.ones_like(vec_norm) * 0, vec_norm)
            z_axis = op_norm
            temp_ae=tf.reduce_sum(tf.multiply(tf.squeeze(mean_local,axis=-2),op_norm),axis=-1,keepdims=True)
            
            temp_ae=tf.multiply(op_norm,temp_ae)
            pipm=tf.squeeze(mean_local,axis=-2)-temp_ae
            
            print(pipm.shape)
            vec_dist = tf.norm(pipm, axis=-1, keepdims =True)
            vec_norm = tf.divide(pipm, vec_dist)
            x_axis= tf.where(tf.is_nan(vec_norm), tf.ones_like(vec_norm) * 0, vec_norm)
            print(x_axis.shape,z_axis.shape)
            y_axis_x=tf.expand_dims(tf.multiply(z_axis[:,:,1],x_axis[:,:,2])-tf.multiply(z_axis[:,:,2],x_axis[:,:,1]),axis=-1)
            y_axis_y=tf.expand_dims(z_axis[:,:,2]*x_axis[:,:,0]-z_axis[:,:,0]*x_axis[:,:,2],axis=-1)
            y_axis_z=tf.expand_dims(z_axis[:,:,0]*x_axis[:,:,1]-z_axis[:,:,1]*x_axis[:,:,0],axis=-1)
            y_axis=tf.concat([y_axis_x,y_axis_y,y_axis_z],axis=-1) 
            x_axis= tf.expand_dims(x_axis,axis=2)  
            y_axis= tf.expand_dims(y_axis,axis=2)
            z_axis= tf.expand_dims(z_axis,axis=2)                    
            x_nn_pts=tf.reduce_sum(tf.multiply(x_axis,nn_pts_local),axis=-1,keepdims=True)
            y_nn_pts=tf.reduce_sum(tf.multiply(y_axis,nn_pts_local),axis=-1,keepdims=True)
            z_nn_pts=tf.reduce_sum(tf.multiply(z_axis,nn_pts_local),axis=-1,keepdims=True)
            if fts_prev_AE is not None:
                nn_fts_AE  =tf.concat([x_nn_pts,y_nn_pts,z_nn_pts,rotate_AE,trans_AE,fts_prev_AE,my_fea],axis=-1)
            else:
                nn_fts_AE  =tf.concat([x_nn_pts,y_nn_pts,z_nn_pts,my_fea],axis=-1)
       
    with tf.variable_scope(tag+'bin_idx_RI', reuse=tf.AUTO_REUSE):
        # compute indices from nn_pts_local_proj
        vec = - nn_pts_center
        vec_dist = tf.norm(vec, axis=-1, keepdims =True)
        vec_norm = tf.divide(vec, vec_dist)
        nn_pts_local_proj = tf.matmul(nn_pts_local, vec_norm, transpose_b=True)

        proj_min = tf.reduce_min(nn_pts_local_proj, axis=-2, keepdims=True) 
        proj_max = tf.reduce_max(nn_pts_local_proj, axis=-2, keepdims=True) 
        seg = (proj_max - proj_min) / D

        vec_tmp = tf.range(0, D, 1, dtype=tf.float32)
        vec_tmp = tf.reshape(vec_tmp, (1,1,1,D))

        limit_bottom = vec_tmp * seg + proj_min
        limit_up = limit_bottom + seg

        idx_up = nn_pts_local_proj <= limit_up
        idx_bottom = nn_pts_local_proj >= limit_bottom
        idx = tf.to_float(tf.equal(idx_bottom, idx_up))
        idx_expand = tf.expand_dims(idx, axis=-1)
    
    with tf.variable_scope(tag+'conv_RI_1', reuse=tf.AUTO_REUSE):
        [N,P,K,dim] = nn_fts_RI.shape # (N, P, K, 3)
        
        C_pts_fts = 64
        nn_fts_local_reshape = tf.reshape(nn_fts_RI, (-1,P*K,dim,1))
        nn_fts_local = tf_util.conv2d(nn_fts_local_reshape, C_pts_fts//2, [1,dim],
                             padding='VALID', stride=[1,1], 
                             bn=True, is_training=is_training,
                             scope=tag+'conv_pts_fts_0', bn_decay=bn_decay)
        nn_fts_local = tf_util.conv2d(nn_fts_local, C_pts_fts, [1,1],
                             padding='VALID', stride=[1,1], 
                             bn=True, is_training=is_training,
                             scope=tag+'conv_pts_fts_1', bn_decay=bn_decay)
        nn_fts_local_RI = tf.reshape(nn_fts_local, (-1,P,K,C_pts_fts))
    with tf.variable_scope(tag+'conv_AE_1', reuse=tf.AUTO_REUSE):
        [N,P,K,dim] = nn_fts_AE.shape # (N, P, K, 3)
        
        C_pts_fts = 64
        nn_fts_local_reshape = tf.reshape(nn_fts_AE, (-1,P*K,dim,1))
        nn_fts_local = tf_util.conv2d(nn_fts_local_reshape, C_pts_fts//2, [1,dim],
                             padding='VALID', stride=[1,1], 
                             bn=True, is_training=is_training,
                             scope=tag+'conv_pts_fts_0', bn_decay=bn_decay)
        nn_fts_local = tf_util.conv2d(nn_fts_local, C_pts_fts, [1,1],
                             padding='VALID', stride=[1,1], 
                             bn=True, is_training=is_training,
                             scope=tag+'conv_pts_fts_1', bn_decay=bn_decay)
        nn_fts_local_AE = tf.reshape(nn_fts_local, (-1,P,K,C_pts_fts))
    with tf.variable_scope(tag+'fts_prev_RI', reuse=tf.AUTO_REUSE):
        if fts_prev_RI is not None:
            fts_prev_RI = tf.gather_nd(fts_prev_RI, indices, name=tag + 'fts_prev_RI')  # (N, P, K, 3)
            pts_X_RI = tf.concat([nn_fts_local_RI,fts_prev_RI], axis=-1)
        else:
            pts_X_RI = nn_fts_local_RI
            
    
    
    
        
    with tf.variable_scope(tag+'bin', reuse=tf.AUTO_REUSE):
        pts_X_RI_expand = tf.expand_dims(pts_X_RI, axis=-2)
    
        nn_fts_rect_RI = pts_X_RI_expand * idx_expand
    
        idx_RI = tf.to_float(nn_fts_rect_RI == 0.0)
    
        nn_fts_rect_RI = nn_fts_rect_RI + idx_RI*(-99999999999.0)
    
        nn_fts_rect_RI = tf.reduce_max(nn_fts_rect_RI, axis=-3)
    
    if use_fea_nn or fts_prev_AE is not None:
        nn_fts_rect_AE=tf.concat([nn_fts_local_AE,trans_AE],axis=-1)
    else:
        nn_fts_rect_AE=nn_fts_local_AE
    fts_X_RI = tf_util.conv2d(nn_fts_rect_RI, C, [1,nn_fts_rect_RI.shape[-2].value],
                             padding='VALID', stride=[1,1], 
                             bn=True, is_training=is_training,
                             scope=tag+'conv_RI_2', bn_decay=bn_decay)
    
    
    fts_X_AE = tf_util.conv2d(nn_fts_rect_AE, C, [1,nn_fts_rect_AE.shape[-2].value],
                             padding='VALID', stride=[1,1], 
                             bn=True, is_training=is_training,
                             scope=tag+'conv_AE_2', bn_decay=bn_decay)
    
    
    
    with tf.variable_scope(tag+'global_cat', reuse=tf.AUTO_REUSE): 
        glb_fts_RI=tf.tile(tf.reduce_max(fts_X_RI, axis=1, keepdims=True),[1,P,1,1])
        fts_X_RI=tf.concat([fts_X_RI,glb_fts_RI],axis=-1)
        if use_fea_nn:
            glb_fts_AE=tf.tile(tf.reduce_max(fts_X_AE, axis=1, keepdims=True),[1,P,1,1])
            fts_X_AE=tf.concat([fts_X_AE,glb_fts_AE],axis=-1)
        
        
    
    fts_X_RI = tf_util.conv2d(fts_X_RI, C*2, [1,1],
                              padding='VALID', stride=[1,1], 
                              bn=True, is_training=is_training,
                              scope=tag+'conv_RI_3', bn_decay=bn_decay)
    fts_X_AE = tf_util.conv2d(fts_X_AE, C*2, [1,1],
                              padding='VALID', stride=[1,1], 
                              bn=True, is_training=is_training,
                              scope=tag+'conv_AE_3', bn_decay=bn_decay)
    fea_RI=tf.squeeze(fts_X_RI, axis=-2)
    fea_AE=tf.squeeze(fts_X_AE, axis=-2)
    if use_fea_nn:
        return fea_MLP, fea_RI, fea_AE
    else:
        return fea_MLP, fea_RI, fea_AE, tf.concat([x_axis,y_axis,z_axis],axis=2)
def diff_att(fea1,fea2,fea3,target_score,tag,is_training):
    with tf.variable_scope(tag+'att', reuse=tf.AUTO_REUSE):
        #features=tf.concat([tf.expand_dims(features,axis=2),tf.expand_dims(layer_fts,axis=2),tf.expand_dims(EAfea,axis=2)],axis=2)
        mean_fea=tf.reduce_mean(tf.concat([tf.expand_dims(fea1,axis=2),tf.expand_dims(fea2,axis=2),tf.expand_dims(fea3,axis=2)],axis=2),axis=2)
        minus1=fea1-mean_fea
        minus2=fea2-mean_fea
        minus3=fea3-mean_fea
        with tf.variable_scope(tag+'score1', reuse=tf.AUTO_REUSE):
            score1=mlp_conv(minus1,[32,1])
        with tf.variable_scope(tag+'score2', reuse=tf.AUTO_REUSE):
            score2=mlp_conv(minus2,[32,1])
        with tf.variable_scope(tag+'score3', reuse=tf.AUTO_REUSE):
            score3=mlp_conv(minus3,[32,1])
        minus=tf.concat([tf.expand_dims(minus1,axis=-1),tf.expand_dims(minus2,axis=-1),tf.expand_dims(minus3,axis=-1)],axis=-1)
        score=tf.expand_dims(tf.nn.softmax(tf.concat([score1,score2,score3],axis=-1),axis=-1),axis=-2)
        mean_score=tf.reduce_mean(score,axis=1)
        avg_loss=mean_score-target_score
        avg_loss=tf.reduce_mean(tf.abs(avg_loss))                                
        score=tf.tile(score,[1,1,fea1.shape[-1],1])
        res=tf.reduce_sum(tf.multiply(minus,score),axis=-1)
        features=res+mean_fea
        return features,avg_loss

class Model:
    def __init__(self, inputs, gt1, gt2, gt, target_score, is_training):
        #self.coarse,self.fine = self.create_encoder(inputs, is_training)
        
        self.fine, self.avg_loss = self.create_encoder(inputs, target_score, is_training)
        #print(self.label.shape,self.fine.shape)
        self.outputs = self.fine
        #self.loss, self.chloss= self.create_loss(self.coarse,self.fine, gt)
        self.loss, self.chloss= self.create_loss(self.fine, gt,self.avg_loss)          
        self.visualize_ops = [inputs, self.fine, gt]
        self.visualize_titles = ['input', 'fine output', 'ground truth']        

    def create_encoder(self,layer_pts, target_score, is_training):
        with tf.variable_scope('L1', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('fps', reuse=tf.AUTO_REUSE):
                index=farthest_point_sample(256, layer_pts)
                qrs = gather_point(layer_pts, index)
            fea_MLP, fea_RI, fea_AE, rot=sub_module(layer_pts, None, None, None, qrs, None, None, is_training, 'main', 64, 4, 256, 128, False)
            fea,avg_loss1=diff_att(fea_MLP, fea_RI, fea_AE, target_score,'att',is_training)
            layer_pts = qrs    
        with tf.variable_scope('L2', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('fps', reuse=tf.AUTO_REUSE):
                index=farthest_point_sample(128, layer_pts)
                qrs = gather_point(layer_pts, index)
                qrs_fea = gather_point(fea, index)
                qrs_rot = gather_point(tf.reshape(rot,shape=(rot.shape[0],rot.shape[1],-1)), index)
            fea_MLP, fea_RI, fea_AE, rot=sub_module(layer_pts, fea, fea, rot, qrs, qrs_fea, qrs_rot, is_training, 'main', 32, 2, 128, 256, False)
            fea,avg_loss2=diff_att(fea_MLP, fea_RI, fea_AE, target_score,'att',is_training)                                                
            layer_pts = qrs
        with tf.variable_scope('L3', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('fps', reuse=tf.AUTO_REUSE):
                index=farthest_point_sample(64, layer_pts)
                qrs = gather_point(layer_pts, index)
                qrs_fea = gather_point(fea, index)
                qrs_rot = gather_point(tf.reshape(rot,shape=(rot.shape[0],rot.shape[1],-1)), index) 
            fea_MLP, fea_RI, fea_AE=sub_module(layer_pts, fea, fea, rot, qrs, qrs_fea, qrs_rot, is_training, 'main', 16, 1, 64, 512, True)
            fea,avg_loss3=diff_att(fea_MLP, fea_RI, fea_AE, target_score, 'att',is_training)      
        layer_fts_global=tf.reduce_max(fea, axis=1)
        avg_loss=(0.5*avg_loss2+1.0*avg_loss3)/tf.constant(1.5)
        
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            outputs = mlp(layer_fts_global, [1024, 1024, 16384 * 3])
            outputs = tf.reshape(outputs, [-1, 16384, 3])

        return outputs,avg_loss

    
    def create_loss(self, fine , gt, avg_loss):
        loss3=chamfer(fine, gt)
        
        loss=0.01*avg_loss+loss3
        
        return loss, loss3
        
