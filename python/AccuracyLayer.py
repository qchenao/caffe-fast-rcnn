import caffe
import numpy as np
from copy import deepcopy

class AccuracyViewLayer(caffe.Layer):

    def setup(self, bottom, top):

        if bottom[0].num != bottom[1].num:
            raise Exception("The data and label should have the same number.")


        params = eval(self.param_str_)
        self.tol_angle = params.get('tol_angle', 5)
        self.period = params.get('period', 360)

    def reshape(self, bottom, top):


        if (bottom[1].count / bottom[1].num != self.period * 12 ):
            raise Exception("number of classes == 4320.")

        top[0].data.reshape(1,1,1,1)


    def forward(self, bottom, top):
        # assign output
        accuracy = 0

        data = deepcopy(bottom[0].data[:])
        label = deepcopy(bottom[1].data[:])
        label[label >= 10000] -= 10000
        angle = np.array(label % (self.period))
        cls_idx = label / (self.period)
        nonbkg_cnt = cls_idx.count(12)

         = np.zeros(bottom[0].num,self.period)

        for i in range(bottom[0].num):
            if sls_idx == 12:
                continue
            tmp_angle = data[i][np.arange(cls_idx[i]*self.period,(cls_idx[i]+1)*self.period)]
            pred_angle = np.argmax(tmp_angle)
            error = min(abs(pred_angle - angle), self.period - abs(pred_angle - angle))
            accuracy += (error <= tol_angle_)

    def backward(self, top, propagate_down, bottom):
        pass
