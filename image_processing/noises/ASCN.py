import numpy as np
import math as m
import cv2


def __ascn2D_fft_gen(AWGN, gsigma):
    s = AWGN.shape


    x = range(int(-s[1] / 2), int(s[1] / 2))
    y = range(int(-s[0] / 2), int(s[0] / 2))
    xgrid, ygrid = np.meshgrid(x, y)
    size = xgrid.shape
    G = np.zeros(size)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            G[i, j] = m.exp(-m.pi * (xgrid[i, j] ** 2 + ygrid[i, j] ** 2) / (2 * gsigma ** 2))
    g = np.fft.fft2(G)
    n = np.fft.fft2(AWGN)


    ASCN = np.fft.ifft2(g * n)
    ASCN = ASCN / np.std(ASCN)
    return ASCN


def ASCN(image, nsigma=0, gsigma=1):
    nimg = cv2.resize(image, (512, 512))
    im_shape = nimg.shape
    nimg[:, :, 0] += np.uint8(nsigma * __ascn2D_fft_gen(np.random.randn(im_shape[0], im_shape[1]), gsigma))
    nimg[:, :, 1] += np.uint8(nsigma * __ascn2D_fft_gen(np.random.randn(im_shape[0], im_shape[1]), gsigma))
    nimg[:, :, 2] += np.uint8(nsigma * __ascn2D_fft_gen(np.random.randn(im_shape[0], im_shape[1]), gsigma))

    return nimg
