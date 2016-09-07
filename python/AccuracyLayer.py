import caffe
import numpy as np
from copy import deepcopy

class AccuracyView(caffe.Layer):

    def setup(self, bottom, top):


        if bottom[0].num != bottom[1].num:
            raise Exception("The data and label should have the same number.")


        params = eval(self.param_str_)
        self.tol_angle = params.get('tol_angle', 5)
        self.period = int(params.get('period', 360))


    def reshape(self, bottom, top):

        if (bottom[0].data.size / bottom[0].num != self.period * 12 ):
            raise Exception("number of classes != 4320.")
        top[0].reshape(1)



    def forward(self, bottom, top):

        # assign output
        accuracy = float(0)

        data = np.array(bottom[0].data)
        label = np.array(bottom[1].data).reshape(bottom[0].num).astype(int)
        label[label >= 10000] -= 10000
        angle = np.array(label % (self.period))
        cls_idx = label / (self.period)
        nonbkg_cnt = np.count_nonzero(cls_idx!=12)

         #= np.zeros(bottom[0].num,self.period)

        for i in range(bottom[0].num):
            if cls_idx[i] == 12:
                continue
            tmp_angle = data[i][np.arange(cls_idx[i]*self.period,(cls_idx[i]+1)*self.period)]
            pred_angle = np.argmax(tmp_angle)
            #print pred_angle,'vs',angle[i]
            error = min(abs(pred_angle - angle[i]), self.period - abs(pred_angle - angle[i]))
            #print 'error', error
            accuracy += (error <= self.tol_angle)
        top[0].data[0] = accuracy / nonbkg_cnt

    def backward(self, top, propagate_down, bottom):
        pass
