import caffe
import lmdb
import numpy as np
#import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
lmdb_file = '../data/ShapeNet/syn_lmdb_train_label'

def Read_Render4CNN(lmdb_file, index):

    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin(buffers=True)
    datum = caffe_pb2.Datum()
    data = []
    for ind in index:
        print 'ind',ind
        buf = lmdb_txn.get('%010d'%ind)
        datum.ParseFromString(bytes(buf))
        tmp = caffe.io.datum_to_array(datum)
        data = np.append(data,tmp)
        #im = data.astype(np.uint8)
        #im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
        #print "index ", ind
        #print "data", data

    return data
    #lmdb_cursor = lmdb_txn.cursor()
data = Read_Render4CNN(lmdb_file, [0,402687-1,402687, 529866-1, 529866, 708877-1, 708877, 888483-1, 888483, 1067546-1, 1067546, 1242939, 1242939-1, 1419838-1, 1419838, 1595969-1, 1595969, 1775589-1, 1775589-1,1775589,1955446-1, 1955446, 2135152-1,2135152])
print data

    #for key, value in lmdb_cursor:
    #     datum.ParseFromString(value)
    #
    #     index = datum.label
    #     data = caffe.io.datum_to_array(datum)
    #     #im = data.astype(np.uint8)
    #     #im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
    #     print "index ", index
    #     print "data", data
    #     ##plt.imshow(im)
    #     #plt.show()
