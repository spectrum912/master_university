import copy

import cv2
import os

import numpy as np

'''For image distortion'''
from image_processing.noises import AWGN, ASCN, Mult, Speckle

'''End'''
'''For image filtering'''
from image_processing.filters import DCT, Frost, Lee, Median
# from ksvd import ApproximateKSVD
from image_processing.filters.KSVD import ksvd
from image_processing.filters.BM3D_color import BM3D_1st_step_color, BM3D_2nd_step_color
from scipy import ndimage

'''End'''
'''For image metric'''
from image_processing.metrics.PSNR import psnr
from image_processing.metrics.FSIM import fsim
from psnr_hvsm import psnr_hvs_hvsm

'''End'''

cv2.setUseOptimized(True)


def metric(original, noised):
    gray_img_load = cv2.cvtColor(np.float32(original), cv2.COLOR_BGR2GRAY)
    gray_img_noise = cv2.cvtColor(np.float32(noised), cv2.COLOR_BGR2GRAY)
    psnr_return = psnr(original, noised)
    fsim_return = fsim(original, noised)
    psnr_hvsm_return = psnr_hvs_hvsm(gray_img_load / 255, gray_img_noise / 255)[1]
    return [f"PSNR {psnr_return}\nFSIM {fsim_return}\nPSNR-HVS-M {psnr_hvsm_return}",
            psnr_return, fsim_return, psnr_hvsm_return]


if __name__ == "__main__":
    result_path = "test_result"
    noise_type = "AWGN"
    filter_type = "KSVD"

    image_path = "image_Lena512rgb.png"
    image_loaded = cv2.imread(image_path)

    if noise_type == "AWGN":
        img_noise = AWGN.AWGN(image_loaded, 20)  # .astype(int)
    elif noise_type == "ASCN":
        img_noise = ASCN.ASCN(image_loaded, 10, 0.4)
    elif noise_type == "MULT":
        img_noise = Mult.Mult(image_loaded, 5)
    elif noise_type == "SPECKLE":
        img_noise = Speckle.Speckle(image_loaded, 4, 0.5)
    elif noise_type == "COPY":
        img_noise = copy.copy(image_loaded)

    if noise_type != "":
        cv2.imwrite(f"{result_path}/{noise_type}_Noise.png", img_noise)

    if filter_type == "BM3D":
        print("BM3D")
        imgYCB = cv2.cvtColor(img_noise.astype(np.uint8), cv2.COLOR_BGR2YCrCb)

        Basic_img = BM3D_1st_step_color(imgYCB)
        cv2.imwrite(f"{result_path}/{noise_type}_Basic_sigma_color.png", cv2.cvtColor(Basic_img, cv2.COLOR_YCrCb2BGR))

        print(metric(image_loaded, Basic_img)[0])

        Final_img = BM3D_2nd_step_color(Basic_img, imgYCB)
        cv2.imwrite(f"{result_path}/{noise_type}_Final_sigma_color.png", cv2.cvtColor(Final_img, cv2.COLOR_YCrCb2BGR))
        print(metric(image_loaded, Final_img)[0])

    elif filter_type == "DCT":
        print("DCT")
        img = DCT.dct_filter(img_noise, 100)
        cv2.imwrite(f"{result_path}/{noise_type}_DCT_color.png", img)
        print(metric(image_loaded, img)[0])

    elif filter_type == "FROST":
        print("Frost")
        img = Frost.Frost(img_noise)
        cv2.imwrite(f"{result_path}/{noise_type}_Frost.png", img)
        print(metric(image_loaded, img)[0])

    elif filter_type == "LEE":
        print("Lee")
        img = Lee.lee_filter(img_noise, 5, 20 ** 2)
        cv2.imwrite(f"{result_path}/{noise_type}_Lee.png", Lee.lee_filter(img, 5, 20 ** 2))
        print(metric(image_loaded, img)[0])

    elif filter_type == "MEDIAN":
        print("Median")
        img = Median.median_filter(img_noise, 3)
        cv2.imwrite(f"{result_path}/{noise_type}_Median.png", img)
        print(metric(image_loaded, img)[0])

    elif filter_type == "KSVD":
        print("K-SVD")
        ksvd(f"{result_path}/{noise_type}_Noise.png", f"{result_path}/{noise_type}_K-SVD.png")
        img = cv2.imread(f"{result_path}/{noise_type}_K-SVD.png")
        print(metric(image_loaded, img)[0])
