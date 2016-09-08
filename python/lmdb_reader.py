import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
# Wei Yang 2015-08-19
# Source
#   Read LevelDB/LMDB
#   ==================
#       http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html
#   Plot image
#   ==================
#       http://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/
#   Creating LMDB in python
#   ==================
#       http://deepdish.io/2015/04/28/creating-lmdb-in-python/


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
