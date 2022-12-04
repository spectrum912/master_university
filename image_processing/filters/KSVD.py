# coding: utf-8
import cv2
import numpy as np
from skimage import io, util
from sklearn.feature_extraction import image
from ksvd import ApproximateKSVD


def clip(img):
    img = np.minimum(np.ones(img.shape), img)
    img = np.maximum(np.zeros(img.shape), img)
    return img


def ksvd(input_img):
    cv2.imwrite("temp_img.png", input_img)
    img = util.img_as_float(io.imread("temp_img.png"))
    patch_size = (5, 5)
    patches = image.extract_patches_2d(img, patch_size)
    signals = patches.reshape(patches.shape[0], -1)
    mean = np.mean(signals, axis=1)[:, np.newaxis]
    signals -= mean
    aksvd = ApproximateKSVD(n_components=32)
    dictionary = aksvd.fit(signals[:10000]).components_
    gamma = aksvd.transform(signals)
    reduced = gamma.dot(dictionary) + mean
    reduced_img = image.reconstruct_from_patches_2d(
        reduced.reshape(patches.shape), img.shape)
    io.imsave("temp_img.png", clip(reduced_img))
    return cv2.imread("temp_img.png")

# import dictlearn as dl
# import matplotlib.pyplot as plt
# from scipy import misc
#
# # Set default pyplot colormat
# plt.rcParams['image.cmap'] = 'bone'
#
# clean = misc.imread('images/lena512.png').astype(float)
# noisy = misc.imread('images/lena_noisy512.png').astype(float)
#
# denoiser = dl.Denoise(noisy, patch_size=10, method='batch')
# denoiser.train(iters=40, n_nonzero=1, n_atoms=256, n_threads=4)
# denoised = denoiser.denoise(sigma=33, n_threads=4)
#
# plt.subplot(131)
# plt.imshow(clean)
# plt.axis('off')
# plt.title('Clean')
#
# plt.subplot(132)
# plt.imshow(noisy)
# plt.axis('off')
# plt.title('Noisy, psnr = {}'.format(dl.utils.psnr(clean, noisy, 255)))
#
# plt.subplot(133)
# plt.imshow(denoised)
# plt.axis('off')
# plt.title('Denoised, psnr = {}'.format(dl.utils.psnr(clean, denoised, 255)))
# plt.show()
