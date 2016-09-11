import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from multiprocessing.dummy import Pool as ThreadPool
import time

def read_lmdb(lmdb_file, index):

    pool = ThreadPool(4)
    read_para = Read_Render4CNN(lmdb_file)
    #print read_para(0).shape
    start_time = time.time()
    results = pool.map(read_para, index)
    elapsed_time = time.time() - start_time
    pool.close()
    pool.join()
    return results

class Read_Render4CNN(object):

  def __init__(self, lmdb_file):
    self.lmdb_env = lmdb.open(lmdb_file)
  def __call__(self, ind):
    return Read_Once(ind, self.lmdb_env)

def Read_Once(ind, lmdb_env):

    lmdb_txn = lmdb_env.begin(buffers=True)
    datum = caffe_pb2.Datum()
    buf = lmdb_txn.get('%010d'%ind)
    datum.ParseFromString(bytes(buf))
    data = caffe.io.datum_to_array(datum)
    #print 'ind', ind, data.shape
    return data
