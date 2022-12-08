import copy

import cv2
import os

import numpy as np

'''For image distortion'''
from image_processing.noises import AWGN, ASCN, Mult, Speckle

'''End'''
'''For image filtering'''
from image_processing.filters import DCT, Frost, Lee, Median
from image_processing.filters.BM3D_color import BM3D_1st_step_color
from image_processing.filters.KSVD import ksvd

'''End'''
'''For image metric'''

'''End'''

from concurrent.futures import ProcessPoolExecutor as Executor


image_path = 'dataset/coco/train2017'
processed_path = 'E:/diplom'

cv2.setUseOptimized(True)

dirs_noise = ["AWGN_5", "AWGN_10", "AWGN_20", "AWGN_30", "MULT_1",
              "MULT_3", "MULT_5", "MULT_7", "Speckle_1", "Speckle_5"]
dirs_filter = ["BM3D", "DCT", "Frost", "Lee", "Median", "KSVD"]


def make_dirs():
    for df in dirs_filter:
        path = processed_path + '\\filtered\\' + df
        if not os.path.exists(path):
            os.mkdir(path)
        for dn in dirs_noise:
            path = processed_path + '\\filtered\\' + df + '\\' + dn
            if not os.path.exists(path):
                os.mkdir(path)

    for dn in dirs_noise:
        path = processed_path + '\\noised\\' + dn
        if not os.path.exists(path):
            os.mkdir(path)


def processing_noise(image):

    image_loaded = cv2.imread(f"{image_path}/{image}")

    img_noise = AWGN.AWGN(image_loaded, 5)  # .astype(int)
    cv2.imwrite(f"{processed_path}/noised/AWGN_5/{image}", img_noise)

    img_noise = AWGN.AWGN(image_loaded, 10)  # .astype(int)
    cv2.imwrite(f"{processed_path}/noised/AWGN_10/{image}", img_noise)

    img_noise = AWGN.AWGN(image_loaded, 20)  # .astype(int)
    cv2.imwrite(f"{processed_path}/noised/AWGN_20/{image}", img_noise)

    img_noise = AWGN.AWGN(image_loaded, 30)  # .astype(int)
    cv2.imwrite(f"{processed_path}/noised/AWGN_30/{image}", img_noise)

    # img_noise = ASCN.ASCN(image_loaded, 5, 0.8)
    # cv2.imwrite(f"{processed_path}/noised/ASCN_30/{image}", img_noise)

    img_noise = Mult.Mult(image_loaded, 1)  # ли, фрост, медиан
    cv2.imwrite(f"{processed_path}/noised/MULT_1/{image}", img_noise)

    img_noise = Mult.Mult(image_loaded, 3)
    cv2.imwrite(f"{processed_path}/noised/MULT_3/{image}", img_noise)

    img_noise = Mult.Mult(image_loaded, 5)
    cv2.imwrite(f"{processed_path}/noised/MULT_5/{image}", img_noise)

    img_noise = Mult.Mult(image_loaded, 7)
    cv2.imwrite(f"{processed_path}/noised/MULT_7/{image}", img_noise)

    img_noise = Speckle.Speckle(image_loaded, 1, 1)
    cv2.imwrite(f"{processed_path}/noised/Speckle_1/{image}", img_noise)

    img_noise = Speckle.Speckle(image_loaded, 5, 1)
    cv2.imwrite(f"{processed_path}/noised/Speckle_5/{image}", img_noise)


def processing_filter(typeOfNoise):

    for img_noise in os.listdir(f"{processed_path}/noised/{typeOfNoise}"):

        print(img_noise)

        if not os.path.exists(f"{processed_path}/filtered/BM3D/{typeOfNoise}/{img_noise}"):
            img_read = cv2.imread(f"{processed_path}/noised/{typeOfNoise}/{img_noise}")
            imgYCB = cv2.cvtColor(img_read.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
            Basic_img = BM3D_1st_step_color(imgYCB)
            cv2.imwrite(f"{processed_path}/filtered/BM3D/{typeOfNoise}/{img_noise}", cv2.cvtColor(Basic_img, cv2.COLOR_YCrCb2BGR))

        if not os.path.exists(f"{processed_path}/filtered/DCT/{typeOfNoise}/{img_noise}"):
            img_read = cv2.imread(f"{processed_path}/noised/{typeOfNoise}/{img_noise}")
            img = DCT.dct_filter(img_read, 100)
            cv2.imwrite(f"{processed_path}/filtered/DCT/{typeOfNoise}/{img_noise}", img)

        if not os.path.exists(f"{processed_path}/filtered/Frost/{typeOfNoise}/{img_noise}"):
            img_read = cv2.imread(f"{processed_path}/noised/{typeOfNoise}/{img_noise}")
            img = Frost.Frost(img_read)
            cv2.imwrite(f"{processed_path}/filtered/Frost/{typeOfNoise}/{img_noise}", img)

        if not os.path.exists(f"{processed_path}/filtered/Lee/{typeOfNoise}/{img_noise}"):
            img_read = cv2.imread(f"{processed_path}/noised/{typeOfNoise}/{img_noise}")
            img = Lee.lee_filter(img_read, 5, 20 ** 2)
            cv2.imwrite(f"{processed_path}/filtered/Lee/{typeOfNoise}/{img_noise}", Lee.lee_filter(img, 5, 20 ** 2))

        if not os.path.exists(f"{processed_path}/filtered/Median/{typeOfNoise}/{img_noise}"):
            img_read = cv2.imread(f"{processed_path}/noised/{typeOfNoise}/{img_noise}")
            img = Median.median_filter(img_read, 5)
            cv2.imwrite(f"{processed_path}/filtered/Median/{typeOfNoise}/{img_noise}", img)

        if not os.path.exists(f"{processed_path}/filtered/KSVD/{typeOfNoise}/{img_noise}"):
            img_read = cv2.imread(f"{processed_path}/noised/{typeOfNoise}/{img_noise}")
            img = ksvd(img_read)
            cv2.imwrite(f"{processed_path}/filtered/KSVD/{typeOfNoise}/{img_noise}", img)



if __name__ == '__main__':
    # make_dirs()
    # _________________NOISE___________________________________________________
    # POOL_SIZE = os.cpu_count() - 2
    # with Executor(max_workers=POOL_SIZE) as executor:
    #     executor.map(processing_noise, os.listdir(path=os.path.join(image_path)))
    # _________________NOISE___________________________________________________

    # _________________FILTER___________________________________________________

    print("start")
    im_path = f"{processed_path}/noised"
    POOL_SIZE = os.cpu_count() - 1

    with Executor(max_workers=POOL_SIZE) as executor:
        try:
            for r in executor.map(processing_filter, os.listdir(im_path)):
                try:
                    print(r)
                except Exception as exc:
                    print(f'Catch inside: {exc}')
        except Exception as exc:
            print(f'Catch outside: {exc}')

    # _________________FILTER___________________________________________________

    # print("start")
    # im_path = f"{processed_path}/noised"
    # i = 0
    # for f in os.listdir(path=os.path.join(im_path)):
    #     i += 1
    #     print(i)
    #     processing_filter(f)
