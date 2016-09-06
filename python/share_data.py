#from mnist import *
import numpy as np
# MNIST = mnist()
# ims_training,labels_training = MNIST.load_training()
# ims_testing,labels_testing = MNIST.load_testing()

data_loss=np.array([])
count=0
flag=0
idx_pool=np.array([])

imgnet_mean=np.ndarray(shape=(3,227,227),dtype=int)
imgnet_mean[0] = 104
imgnet_mean[1] = 117
imgnet_mean[2] = 123

Render4CNN_Ind = np.random.randint(0,2314400,size=2314401)

state = 1
iters = 0
