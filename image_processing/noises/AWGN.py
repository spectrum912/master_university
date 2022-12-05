import numpy as np
import cv2
import os
from random import randint


def AWGN(image, sigma=0, mu=0):
    if sigma == 0 and mu == 0:
        return image
    im_shape = image.shape
    noise = np.random.normal(mu, sigma, im_shape)
    noised = np.ndarray.astype(image, np.float) + noise
    r = randint(0, 10000)
    cv2.imwrite(f"{r}.png", noised)
    noised = cv2.imread(f"{r}.png")
    os.remove(f"{r}.png")

    return noised
