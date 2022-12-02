import numpy as np
import math as m


def __ascn2D_fft_gen(AWGN, gsigma):
    s = AWGN.shape
    x = range(int(-s[1] / 2), int(s[1] / 2))
    y = range(int(-s[0] / 2), int(s[0] / 2))
    xgrid, ygrid = np.meshgrid(x, y)
    size = xgrid.shape
    G = np.zeros(size)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            G[i, j] = m.exp(-m.pi * (xgrid[i, j]**2 + ygrid[i, j]**2)/(2 * gsigma**2))
    g = np.fft.fft2(G)
    n = np.fft.fft2(AWGN)
    ASCN = np.fft.ifft2(g*n)
    ASCN = ASCN / np.std(ASCN)
    return ASCN


def ASCN(image, nsigma=0, gsigma=1):
    im_shape = image.shape
    nimg = image + (nsigma * __ascn2D_fft_gen(np.random.randn(im_shape[0], im_shape[1]), gsigma))
    return np.uint8(nimg)