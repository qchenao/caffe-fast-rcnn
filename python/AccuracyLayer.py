import caffe
import numpy as np
import share_data as sd

class AccuracyView(caffe.Layer):

    def setup(self, bottom, top):

        if bottom[0].num != bottom[1].num:
            raise Exception("The data and label should have the same number.")
        self.iter = 0
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
            sd.az[cls_idx[i] * int(self.period /self.tol_angle) + angle[i] / int(self.tol_angle)] += 1
            if (error <= self.tol_angle):
                sd.cor_az[cls_idx[i] * int(self.period /self.tol_angle) + angle[i] / int(self.tol_angle)] += 1
                accuracy += 1

            #print 'share_data.mis_col', sd.mis_col

        sd.record = sd.cor_az / (sd.az + 10**(-10))
        sd.az_board = np.argsort(sd.record)
        top[0].data[0] = accuracy / nonbkg_cnt
        with open("az_baseline_acc.txt", "a") as f:
            f.write(str(top[0].data[0]))
            f.write('\n')
        f.close()
        self.iter += 1
        if self.iter == 12000 :
            with open("baseline_record.txt", "a") as f:
                f.write(str(sd.record))
                f.write('\n')
            f.close()

    def backward(self, top, propagate_down, bottom):
        pass

class AccuracyView_active(caffe.Layer):

    def setup(self, bottom, top):

        if bottom[0].num != bottom[1].num:
            raise Exception("The data and label should have the same number.")

        params = eval(self.param_str_)
        self.tol_angle = params.get('tol_angle', 5)
        self.period = int(params.get('period', 360))
        self.threshold = params.get('threshold', 0.9)

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
            sd.az[cls_idx[i] * int(self.period /self.tol_angle) + angle[i] / int(self.tol_angle)] += 1
            if (error <= self.tol_angle):
                sd.cor_az[cls_idx[i] * int(self.period /self.tol_angle) + angle[i] / int(self.tol_angle)] += 1
                accuracy += 1

            #print 'share_data.mis_col', sd.mis_col

        sd.record = sd.cor_az / (sd.az + 10**(-10))
        sd.az_board = np.argsort(sd.record)
        top[0].data[0] = accuracy / nonbkg_cnt
        with open("az_ohem_acc.txt", "a") as f:
            f.write(str(top[0].data[0]))
            f.write('\n')
        f.close()

    def backward(self, top, propagate_down, bottom):
        pass
