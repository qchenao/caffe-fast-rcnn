import caffe
import numpy as np
import random
import os, struct
from array import array
import share_data
from lmdb_reader import Read_Render4CNN
import pdb


class Render4CNNLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.idx = []
        self.data = []
        params = eval(self.param_str_)
        self.source = params['source']
        self.init = params.get('init', True)
        self.seed = params.get('seed', None)
        self.batch_size=params.get('batch_size', 64)

        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define 1 top: data or label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")


        self.iidx = np.arange(self.batch_size)
        self.idx = share_data.Render4CNN_Ind[self.iidx]



    def reshape(self, bottom, top):

        self.data = Read_Render4CNN(self.source,self.iidx)

        if 'image' in self.source:
            self.data = self.data.reshape(self.batch_size,3,227,227)
            self.data -= share_data.imgnet_mean
            top[0].reshape(self.batch_size,3,227,227)
        else:
            top[0].reshape(self.batch_size,4,1,1)
            self.data = self.data.reshape(self.batch_size,4,1,1)



    def forward(self, bottom, top):
        # assign output

        top[0].data[...] = self.data

        #training small amount of data

        self.iidx = (self.iidx + self.batch_size) % 2314401

        #new epoch, shuffle again
        if np.max(self.iidx) < self.batch_size:
            share_data.Render4CNN_Ind = np.random.randint(0,2314400,size=2314401)

        self.idx = share_data.Render4CNN_Ind[self.iidx]








    def backward(self, top, propagate_down, bottom):
        pass


class Render4CNNLayer_hard(caffe.Layer):

    def setup(self, bottom, top):

        self.idx = []
        self.data = []
        params = eval(self.param_str_)
        self.source = params['source']
        self.batch_size=params.get('batch_size', 64)

        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define 1 top: data or label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")


        self.iidx = np.arange(self.batch_size)
        self.idx = share_data.Render4CNN_Ind[self.iidx]



    def reshape(self, bottom, top):

        self.data = Read_Render4CNN(self.source,self.idx)

        if 'image' in self.source:
            self.data = self.data.reshape(self.batch_size,3,227,227)
            self.data -= share_data.imgnet_mean
            top[0].reshape(self.batch_size,3,227,227)
        else:
            top[0].reshape(self.batch_size,4,1,1)
            self.data = self.data.reshape(self.batch_size,4,1,1)
        print 'iidx', self.idx


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data


        self.iidx = (self.iidx + self.batch_size) % 2314401

        #new epoch, shuffle again
        if np.max(self.iidx) < self.batch_size:
            share_data.Render4CNN_Ind = np.random.randint(0,2314400,size=2314401)

        self.idx = share_data.Render4CNN_Ind[self.iidx]






    def backward(self, top, propagate_down, bottom):
        pass


class MySoftmaxLayer_hard(caffe.Layer):

    def setup(self, bottom, top):

        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
	    #params = eval(self.param_str)
        #self.split = params['split']

    def reshape(self, bottom, top):

        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
            #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):

        scores = np.array(bottom[0].data)

        #to make softmax stable
        tmp = np.tile(np.max(scores,axis=1),np.max(bottom[1].data).astype(int)+1)
        tmp = tmp.reshape(scores.T.shape).T
        scores = scores-tmp

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(bottom[0].num),np.array(bottom[1].data,dtype=np.uint16).reshape(bottom[1].num)]+10**(-10))

        data_loss = np.sum(correct_logprobs)/bottom[0].num
        '''
        if self.split is 'training':
            if share_data.flag:
                share_data.data_loss = []
            share_data.data_loss = np.append(share_data.data_loss, correct_logprobs)
	    #print 'loss', share_data.data_loss.shape
        '''

        self.diff[...] = probs
        top[0].data[...] = data_loss


    def backward(self, top, propagate_down, bottom):

        delta = self.diff


        #for i in range(2):
        if propagate_down[1]:
            raise Exception("Layer cannot backprop to label inputs.")
        if propagate_down[0]:
            delta[range(bottom[0].num), np.array(bottom[1].data,dtype=np.uint16).reshape(bottom[1].num)] -= 1
            bottom[0].diff[...] = delta/bottom[0].num





class MNISTLayer_hard(caffe.Layer):

    def setup(self, bottom, top):

        self.idx = []
        self.data = []
        self.label = []

        params = eval(self.param_str)
        self.mnist_dir = params['mnist_dir']
        self.split = params['split']
        #self.idx = np.array(params['idx'])
        self.init = params.get('init', True)
        self.seed = params.get('seed', None)
        self.batch_size=params.get('batch_size', 64)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # make eval deterministic
        #if 'training' not in self.split:
            #self.init = False

        # randomization: seed and pick
        if self.split is 'training':
            #random.seed(self.seed)
            #self.idx = np.random.randint(0,59999,size=self.batch_size)
            self.idx = np.array(range(self.batch_size))
        else:
            #random.seed(self.seed)
            #self.idx = np.random.randint(0,9999,size=self.batch_size)
            self.idx = np.array(range(self.batch_size))



    def reshape(self, bottom, top):

        if self.split is 'training':
            self.data = share_data.ims_training[self.idx]
            self.label = share_data.labels_training[self.idx]
        else:
            self.data = share_data.ims_testing[self.idx]
            self.label= share_data.labels_testing[self.idx]


        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.data.shape[0],1,self.data.shape[1],self.data.shape[2])
        top[1].reshape(self.label.shape[0])

        self.data = self.data.reshape(self.data.shape[0], 1, self.data.shape[1], self.data.shape[2])
        #im = np.array(self.data[0][0],dtype='uint8')
        #img = Image.fromarray(im)
        #img.show()
        self.label = self.label.reshape(self.label.shape[0])


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        # top[0].data = self.data
        # top[1].data = self.label

        # pick next input
        #self.idx=share_data.vars
        if self.split is 'training':
            if share_data.count < 1000:
                self.idx = (self.idx + self.batch_size) % 60000
            else:
                if share_data.flag :
                    if (share_data.count // 1000) % 2:
                        self.idx = np.arange(self.batch_size)
                    else:
                        share_data.idx_pool = np.argsort(share_data.data_loss,axis=-1,kind='quicksort',order=None)
                        share_data.idx_pool = share_data.idx_pool[np.arange(30000,60000)]
                        self.idx = share_data.idx_pool[np.arange(self.batch_size)]
                        #np.save('data_loss',share_data.data_loss)
                        #np.save('idx_pool',share_data.idx_pool)
                else:
                    if  ((share_data.count // 1000) % 2):
                        self.idx = (self.idx + self.batch_size) % 60000
                    else:
		                self.idx = share_data.idx_pool[np.arange(self.batch_size)+ (share_data.count % 500)*self.batch_size]





            #self.idx = (self.idx + self.batch_size) % 60000
            #print 'training_index'
            #print self.idx
        else:
            random.seed(self.seed)
            #self.idx = np.random.randint(0,9999,size=self.batch_size)
            #print 'testing_index'
            #print self.idx

            self.idx = (self.idx + self.batch_size) % 10000

    def backward(self, top, propagate_down, bottom):
        pass
