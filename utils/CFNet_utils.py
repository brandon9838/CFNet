import numpy as np

def rotate_point_cloud(batch_data1,batch_data2):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle1 = np.random.uniform() * 2 * np.pi
    rotation_angle2 = np.random.uniform() * 2 * np.pi
    rotation_angle3 = np.random.uniform() * 2 * np.pi
    cosval1 = np.cos(rotation_angle1)
    sinval1 = np.sin(rotation_angle1)
    rotation_matrix1 = np.array([[cosval1, 0, sinval1],
                                [0, 1, 0],
                                [-sinval1, 0, cosval1]])
    cosval2 = np.cos(rotation_angle2)
    sinval2 = np.sin(rotation_angle2)
    rotation_matrix2 = np.array([[1, 0, 0],
                                [0, cosval2, -sinval2],
                                [0, sinval2, cosval2]])
    cosval3 = np.cos(rotation_angle3)
    sinval3 = np.sin(rotation_angle3)
    rotation_matrix3 = np.array([[cosval3, -sinval3, 0],
                                [sinval3, cosval3, 0],
                                [0, 0, 1]])
    rotated_data1 = np.zeros(batch_data1.shape, dtype=np.float32)
    rotated_data2 = np.zeros(batch_data2.shape, dtype=np.float32)
    for k in range(batch_data1.shape[0]): 
        shape_pc1 = batch_data1[k, ...]
        rotated_data1[k, ...] = np.dot(np.dot(np.dot(shape_pc1.reshape((-1, 3)), rotation_matrix1), rotation_matrix2), rotation_matrix3)
    for k in range(batch_data2.shape[0]):
        shape_pc2 = batch_data2[k, ...]
        rotated_data2[k, ...] = np.dot(np.dot(np.dot(shape_pc2.reshape((-1, 3)), rotation_matrix1), rotation_matrix2), rotation_matrix3)
    return rotated_data1,rotated_data2

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]
def axis2cor(x,y,z,pcd):
    '''align pcd to reference axes x,y,z'''
    cos=y[0]/((y[0]**2+y[1]**2)**0.5+1e-8)
    sin=y[1]/((y[0]**2+y[1]**2)**0.5+1e-8)
    matz = np.array([[sin,cos, 0],
                     [-cos,sin, 0],
                     [0, 0, 1]])
    x=np.dot(x, matz)
    y=np.dot(y, matz)
    cos=y[2]/((y[1]**2+y[2]**2)**0.5+1e-8)
    sin=y[1]/((y[1]**2+y[2]**2)**0.5+1e-8)
    matx = np.array([[1 , 0, 0],
                     [0 ,sin,-cos],
                     [0 ,cos,sin ]])
    x=np.dot(x, matx)
    y=np.dot(y, matx)
    cos=x[0]/((x[0]**2+x[2]**2)**0.5+1e-8)
    sin=x[2]/((x[0]**2+x[2]**2)**0.5+1e-8)
    maty= np.array([[cos, 0,  -sin],
                    [0  , 1 , 0   ],
                    [sin, 0 , cos ]])
    x=np.dot(x, maty)
    y=np.dot(y, maty)
    temp=np.dot(np.dot(matz, matx),maty)
    return np.matmul(pcd,temp),x,y,np.matmul(z,temp)
def pca_cpu(x):
    '''compute the principal components of point cloud x'''
    b,n,c= x.shape
    mean = np.mean(x,axis=1)
    x_new = x - mean.reshape(b,1,c)
    cov = np.matmul(x_new.transpose(0,2,1),x_new)/(n - 1) 
    e,v=np.linalg.eig(cov)
    v=v.transpose(0,2,1).reshape(-1,3)
    idx = e.argsort(axis=-1)
    temp=(np.arange(b)*3).reshape(b,1)
    idx=(idx+temp).reshape(-1)
    e=e.reshape(-1)[idx].reshape(b,-1)
    v=v[idx].reshape(b,3,3)
    return e,v


def id2idx(ids):
    classdic={
    '02691156':0,
    '02933112':1,
    '02958343':2,
    '03001627':3,
    '03636649':4,
    '04256520':5,
    '04379243':6,
    '04530566':7}
    return classdic[ids[0:8]]
def id2idx_batch(ids):
    classdic={
    '02691156':0,
    '02933112':1,
    '02958343':2,
    '03001627':3,
    '03636649':4,
    '04256520':5,
    '04379243':6,
    '04530566':7}
    idx=[]
    for i in range(len(ids)):
        idx.append(classdic[ids[i][0:8]])
    return np.array(idx)
def idx2class(idx):
    classdic={
    0:'Airplane  ',
    1:'Cabinet   ',             
    2:'Car       ',
    3:'Chair     ',
    4:'Lamp      ',
    5:'Sofa      ',
    6:'Table     ',
    7:'Watercraft'}
    return classdic[idx]
