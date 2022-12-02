import numpy as np


def Mult(image, looks):
    k = 0.8
    im_shape = image.shape
    noise = np.zeros(im_shape)
    for i in range(0, looks):
        noise = noise + np.random.rayleigh(k, im_shape)
    nimg = image * (noise / looks)
    return nimg.astype(np.uint8)