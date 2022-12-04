import numpy as np
from .DCT_utils import adct2, idct2
import copy

def dct_filter(image, sigma, bsize=8, step=1):
    beta = 2.7
    nimg = copy.copy(image)
    for color in range(3):
        image = nimg[:, :, color]
        s = np.shape(image)
        filtered_image = np.zeros(s, dtype=np.float)
        threshold = beta * sigma
        for i in range(0, s[0]-bsize, step):
            for j in range(0, s[1]-bsize, step):
                im_block = image[i:i+bsize, j:j+bsize]
                dct_block = adct2(im_block).reshape((bsize*bsize, 1))
                for z in range(1, bsize*bsize):
                    if abs(dct_block[z]) <= threshold: dct_block[z] = 0
                filtered_image[i:i+bsize, j:j+bsize] = idct2(dct_block.reshape((bsize, bsize)))
        nimg[:, :, color] = filtered_image
    return nimg
