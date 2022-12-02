import copy
import numpy as np


def local_var(block, mean):
    bl = block.reshape((block.shape[0] * block.shape[1], 1))
    var = float(0)
    for i in bl:
        var += pow(i - mean, 2)
    return var/bl.shape[0]


def lee_filter(band, window, var_noise):
    nimg = copy.copy(band)
    for color in range(3):
        band = nimg[:, :, color]
        s = np.shape(band)
        border = int(window/2)
        for i in range(border, s[0] - border):
            for j in range(border, s[1] - border):
                block = band[i-border:i+border+1, j-border:j+border+1]
                LM = np.mean(block, dtype=np.float64)
                LV = np.var(block, dtype=np.float64)
                K = LV/(LV + var_noise)
                pixel = np.uint8(LM + (K * (np.float64(band[i, j]) - LM)))
                band[i, j] = pixel
        nimg[:, :, color] = band
    return nimg