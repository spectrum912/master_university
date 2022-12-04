import numpy as np
import cv2


def AWGN(image, sigma=0, mu=0):
    if sigma == 0 and mu == 0:
        return image
    im_shape = image.shape
    noise = np.random.normal(mu, sigma, im_shape)
    noised = np.ndarray.astype(image, np.float) + noise
    cv2.imwrite("AWGN.png", noised)
    noised = cv2.imread("AWGN.png")

    return noised
