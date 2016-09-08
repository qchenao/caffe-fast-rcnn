import caffe
import lmdb
import numpy as np
#import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2



def Read_Render4CNN():

    lmdb_file ='../data/ShapeNet/syn_lmdb_train_label'
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    print 'datum'
    cls = []
    az = []
    el = []
    t = []

    for key, value in lmdb_cursor:

        datum.ParseFromString(value)

        index = datum.label
        if (index % 1000 == 0):
            print index
        data = caffe.io.datum_to_array(datum)
        #im = data.astype(np.uint8)
        #im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
        cls = np.append(cls, data[0].reshape(1))
        az = np.append(az, data[1].reshape(1))
        el = np.append(el, data[2].reshape(1))
        t = np.append(t, data[3].reshape(1))
        ##plt.imshow(im)
        #plt.show()
    np.save('cls', cls)
    np.save('az', az)
    np.save('el', el)
    np.save('t', t)

Read_Render4CNN()
