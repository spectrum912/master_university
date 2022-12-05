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

'''End'''
'''For image metric'''

'''End'''

from concurrent.futures import ProcessPoolExecutor as Executor

image_path = 'dataset/coco/train2017'
processed_path = 'E:\\diplom\\noised'

cv2.setUseOptimized(True)


def processing_noise(image):
    image_loaded = cv2.imread(f"{image_path}/{image}")

    img_noise = AWGN.AWGN(image_loaded, 5)  # .astype(int)
    cv2.imwrite(f"{processed_path}/AWGN_5/{image}", img_noise)

    img_noise = AWGN.AWGN(image_loaded, 10)  # .astype(int)
    cv2.imwrite(f"{processed_path}/AWGN_10/{image}", img_noise)

    img_noise = AWGN.AWGN(image_loaded, 20)  # .astype(int)
    cv2.imwrite(f"{processed_path}/AWGN_20/{image}", img_noise)

    img_noise = AWGN.AWGN(image_loaded, 30)  # .astype(int)
    cv2.imwrite(f"{processed_path}/AWGN_30/{image}", img_noise)

    # img_noise = ASCN.ASCN(image_loaded, 5, 0.8)
    # cv2.imwrite(f"{processed_path}/ASCN_30/{image}", img_noise)

    img_noise = Mult.Mult(image_loaded, 1)
    cv2.imwrite(f"{processed_path}/MULT_1/{image}", img_noise)

    img_noise = Mult.Mult(image_loaded, 3)
    cv2.imwrite(f"{processed_path}/MULT_3/{image}", img_noise)

    img_noise = Mult.Mult(image_loaded, 5)
    cv2.imwrite(f"{processed_path}/MULT_5/{image}", img_noise)

    img_noise = Mult.Mult(image_loaded, 7)
    cv2.imwrite(f"{processed_path}/MULT_7/{image}", img_noise)

    img_noise = Speckle.Speckle(image_loaded, 1, 1)
    cv2.imwrite(f"{processed_path}/Speckle_1/{image}", img_noise)

    img_noise = Speckle.Speckle(image_loaded, 5, 1)
    cv2.imwrite(f"{processed_path}/Speckle_5/{image}", img_noise)


if __name__ == '__main__':
    POOL_SIZE = os.cpu_count() - 2
    with Executor(max_workers=POOL_SIZE) as executor:
        executor.map(processing_noise, os.listdir(path=os.path.join(image_path)))

# filtering_frost(os.listdir(path=os.path.join(image_path))[0])

# for image in os.listdir(path=os.path.join(image_path)):
#     th = Process(target=filtering, args=(image_path, image))
#     th.start()
#     c -= 1
#     if c == 0:
#         break


# for image in os.listdir(path=os.path.join(image_path)):  # Start image processing loop
#     image_loaded = cv2.imread(f"{image_path}/{image}")
#     image_noised = img = ASCN.ASCN(image_loaded, 5, 0.8)
#     cv2.imwrite(f"{noised_path}/{image}", image_noised)
#     images_c -= 1
#     if images_c == 0:
#         break
# img_noise = AWGN.AWGN(image_loaded, 20)  # .astype(int)
# img = ASCN.ASCN(image_loaded, 10, 0.4)
# img = Mult.Mult(image_loaded, 5)
# img = Speckle.Speckle(image_loaded, 4, 0.5)
# img = copy.copy(image_loaded)
# cv2.imwrite("Noise.png", img_noise)
