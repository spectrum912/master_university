import numpy as np
import math as m


def dct2Dnps_add_est( nimg, Trans2D ):
    bsize = np.size(Trans2D, 1)
    b = np.array(nimg.shape)
    s = np.array(nimg.shape) - bsize + 1
    vDCT = np.zeros((bsize**2, s[0]*s[1]))

    ind = 0
    for i in range(1, s[0]):
        for j in range(1, s[1]):
            bspat = nimg[i:i+bsize, j:j+bsize]
            bdct = np.reshape(Trans2D * bspat * Trans2D.transpose(), (bsize**2))
            vDCT[:,ind] = bdct[:]**2
            ind = ind + 1

    mvDCT = np.mean(vDCT, 1)

    NPS = np.reshape(mvDCT, [bsize, bsize])
    NPS[0, 0] = 0
    NPS = NPS * (bsize**2 - 1) / np.sum(NPS[:])
    for i in range(0, bsize):
        for j in range(0, bsize):
            NPS[i,j] = m.sqrt(NPS[i,j])
    return NPS