from cv2 import resize
import numpy as np
from image_processing.noises.ASCN import __ascn2D_fft_gen
from copy import copy


def Speckle(image, looks, gsigma):

    nimg = resize(image, (512, 512))
    k = 0.8
    for j in range(3):
        im_shape = nimg[:, :, j].shape
        noise = np.zeros(im_shape)
        for i in range(0, looks):

            ascn = __ascn2D_fft_gen(np.random.randn(im_shape[1], im_shape[0]), gsigma)
            C = ascn
            size = C.shape
            B = np.random.rayleigh(k, size)

            CI = np.argsort(C)
            BI = np.argsort(B)

            C[CI] = B[BI]

            noise = noise + np.reshape(C, im_shape)
        nimg[:, :, j] = nimg[:, :, j] * (noise / looks)

    return nimg.astype(np.uint8)
