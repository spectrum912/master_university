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

image_path = 'dataset/Mult_5'
filtered_path_frost = 'dataset/Mult_5_BM3D'

cv2.setUseOptimized(True)


def filtering_frost(image):
    image_loaded = cv2.imread(f"{image_path}/{image}")
    img = BM3D_1st_step_color(image_loaded)
    cv2.imwrite(f"{filtered_path_frost}/{image}", img)


if __name__ == '__main__':
    POOL_SIZE = os.cpu_count() - 2
    with Executor(max_workers=POOL_SIZE) as executor:
        executor.map(filtering_frost, os.listdir(path=os.path.join(image_path)))

filtering_frost(os.listdir(path=os.path.join(image_path))[0])

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
