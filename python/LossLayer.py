import numpy as np
import caffe

class SoftmaxViewLoss(caffe.Layer):

    def setup(self, bottom, top):

        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute loss.")

        self.weights_sum = 0
        self.prob_data = []
        self.label=[]
        self.weight = []
        self.dim = 0
        self.spatial_dim = 0
        self.cls_idx = []
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.weights_sum = np.sum(np.exp(-abs(range(-bandwidth, bandwidth+1))/float(self.sigma)))

        params = eval(self.param_str_)
        self.bandwidth = params.get('bandwidth', 5)
        self.sigma = params.get('sigma', 3)
        self.pos_weight=params.get('pos_weight', 1)
        self.neg_weight=params.get('neg_weight', 0)
        self.period=int(params.get('period', 360))

    def reshape(self, bottom, top):

        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
            #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
	    #pr 'num',bottom[0].num
        self.prob_data = np.array(bottom[0].data)
        self.label = np.array(bottom[1].data)

        num = bottom[0].num
        self.dim = bottom[0].count
        self.spatial_dim = bottom[0].height * bottom[0].width

        if (self.spatial_dim != 1):
            raise Exception("self.spatial_dim != 1")

        weight = np.zeros_like(self.label)
        weight[self.label < 10000] = pos_weight
        weight[self.label >= 10000] = neg_weight
        self.label[self.label >= 10000] -= 10000
        loss = 0
        
        if (self.dim <= self.label):
            raise Exception("self.label value exceeds dimension")


        self.cls_idx = np.array(self.label / (self.period))
        nonbg_ind = self.cls_idx != 12

        probs_cls_data = self.prob_data.reshape(num, 12, 360)

        probs_cls_sum = np.zeros(shape=(num, 1),dtype=np.float)
        probs_cls_sum[nonbg_ind] = np.sum(abs(probs_cls_data[nonbg_ind, self.cls_idx[nonbg_ind]]))
        probs_cls_sum = np.tile(probs_cls_sum, 360).reshape(360, num).T

        prob_cls_data[nonbg_ind, self.cls_idx[nonbg_ind]] /= probs_cls_sum[nonbg_ind]

        diff = np.zeros_like(prob_cls_data)
        diff[nonbg_ind, self.cls_idx[nonbg_ind]] = np.array(prob_cls_data[nonbg_ind, self.cls_idx[nonbg_ind]] * self.weights_sum)
        self.diff[...] = diff * self.weight

        # convert to 360-class self.label
        view_label = self.label % (self.period)
        view_label = np.tile(view_label, 2*bandwidth+1)
        view_label = view_label.reshape(2*bandwidth+1, num).T

        k = np.arange(-bandwidth, bandwidth+1)
        k = np.tile(k, num)
        k = k.reshape(num, 2*bandwidth+1)

        # e.g. view_label+k=-3 --> 357
        view_k = (view_label + k) % self.period

        cls_idx_broad = np.tile (self.cls_idx * self.period, 2*bandwidth+1).reshape(2*bandwidth+1, num).T
        # convert back to 4320-class self.label
        label_value_k = view_k + cls_idx_broad

        tmp_loss = np.zeros_like(probs_cls_data)

        # loss is weighted by exp(-|dist|/sigma)
        tmp_loss[nonbg_ind] -= exp(-abs(k)/float(sigma)) * log(np.max(probs_cls_data[nonbg_ind][label_value_k [nonbg_ind]],10**(-37)));

        self.diff[nonbg_ind][label_value_k [nonbg_ind]] -= exp(-abs(k)/float(sigma)) * weight[nonbg_ind][label_value_k [nonbg_ind]]

        loss = np.sum(np.sum(tmp_loss * weight))

        top[0].data[0] = loss / num / self.spatial_dim

        self.diff[...] = probs_cls_data.reshape(*bottom[0].data.shape) * self.weight_sum / num / self.spatial_dim


    def backward(self, top, propagate_down, bottom):

        #for i in range(2):
        if propagate_down[1]:
            raise Exception("Layer cannot backprop to self.label inputs.")
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff
