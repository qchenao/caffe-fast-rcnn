import numpy as np
class Map :

    def __init__(self):

        self.cls = np.load('./data/ShapeNet/cls.npy')
        self.az = np.load('./data/ShapeNet/az.npy')
        self.el = np.load('./data/ShapeNet/el.npy')
        self.t = np.load('./data/ShapeNet/t.npy')
        self.offset = [0, 402687, 529866, 708877, 888483, 1067546, 1242939, 1419838, 1595969, 1775589, 1955446, 2135152, 2314401]

    def az2ind(self, cls, low, high, num):

        tmp = np.array(self.az[np.arange(self.offset[cls],self.offset[cls+1])])
        tmp = np.where( (low <= tmp) & (tmp <= high) )[0] + self.offset[cls]
        np.random.shuffle(tmp)
        if num < len(tmp) :
            return tmp[range(num)]
        else:
            return tmp

    def el2ind(self, cls, low, high, num):

        tmp = self.el[np.arange(self.offset[cls],self.offset[cls+1])]
        tmp = np.where( (low <= tmp) & (tmp <= high) )[0] + self.offset[cls]
        np.random.shuffle(tmp)
        if num < len(tmp) :
            return tmp[range(num)]
        else:
            return tmp

    def t2ind(self, cls, low, high, num):

        tmp = self.t[np.arange(self.offset[cls],self.offset[cls+1])]
        tmp = np.where( (low <= tmp) & (tmp <= high) )[0] + self.offset[cls]
        np.random.shuffle(tmp)
        if num < len(tmp) :
            return tmp[range(num)]
        else:
            return tmp

#Map = Map()
#print Map.az2ind(0,0,15,10)
