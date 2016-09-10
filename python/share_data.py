#from mnist import *
import numpy as np
from mapping import Map
# MNIST = mnist()
# ims_training,labels_training = MNIST.load_training()
# ims_testing,labels_testing = MNIST.load_testing()

data_loss=np.array([])
count=0
flag=0


imgnet_mean=np.ndarray(shape=(3,227,227),dtype=np.uint8)
imgnet_mean[0] = 104
imgnet_mean[1] = 117
imgnet_mean[2] = 123

Render4CNN_Ind = np.random.randint(0,2314400,size=2314401)

state = 1
iters = 0

cor_az = np.zeros(24*12,dtype=np.int)
az = np.zeros(24*12,dtype=np.int)
az_board = []
record = []

Map = Map()
idx_pool = []
for i in range(288):
    idx_pool = np.append(idx_pool, Map.az2ind(i/24, i*15, (i+1)*15, 19200))
np.random.shuffle(idx_pool)
