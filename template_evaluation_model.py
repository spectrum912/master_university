import cv2
import os

from yolov5 import YOLOv5
from utils import batch_loop
from CNN_benchmark.utils.coco_load_annotation import LoadDataset
import numpy as np
from CNN_benchmark.draw import drawMatrix
from CNN_benchmark.statistical import ConfusionMatrix, IOU
from CNN_benchmark.statistical.IOU import __calc_iou
import datetime
import json

'''For image distortion'''
from image_processing.noises import AWGN, ASCN, Mult, Speckle

'''End'''

'''For image filtering'''
from image_processing.filters import DCT, Frost, Lee, Median
from ksvd import ApproximateKSVD
from image_processing.filters.BM3D_color import BM3D_1st_step_color, BM3D_2nd_step_color

'''End'''

'''For image metric'''
from image_processing.metrics.PSNR import psnr
from image_processing.metrics.psnrhvsm import psnrhvsm
from image_processing.metrics.FSIM import fsim
from image_procesing_test import metric

'''End'''

# Load images from dataset

processed_path = 'E:/diplom'

image_path = "E:/diplom/filtered/BM3D/AWGN_20"  # 'dataset/coco/train2017'  # for coco -> 'dataset/voc/JPEGImages'
set_name = "AWGN_20-BM3D"

dirs_noise = ["AWGN_5", "AWGN_10", "AWGN_20", "AWGN_30", "MULT_1",
              "MULT_3", "MULT_5", "MULT_7", "Speckle_1", "Speckle_5", "ASCN_10"]

dirs_filter = ["BM3D", "DCT", "Frost", "Lee", "Median", "KSVD"]


# for d_f in dirs_filter:
#     image_path = f"{processed_path}/filtered/{d_f}/{d_n}"

for d_n in dirs_noise:
    image_path = f"{processed_path}/noised/{d_n}"
    set_name = d_n
    print(set_name)
    im_processing_path_list = os.listdir(f"{processed_path}/filtered/DCT/Speckle_1")

    # set_name = f"{d_n}-{d_f}"
    # im_processing_path_list = im_p_l = os.listdir(f"{processed_path}/filtered/{d_f}/{d_n}")

    image_list = []

    y_t = []
    y_p = []

    t_b = {}
    p_b = {}
    i_n = 0

    image_batch = 128  # Count of image for one loop
    noised = False
    filtered = False
    dataset, classnames = LoadDataset('dataset/coco/annotations/instances_train2017.json')

    result_file = open("res.txt", 'a')
    '''This path is unique for each model || Load model'''
    model_path = "yolov5s.pt"
    model = YOLOv5(model_path)
    '''End unique path'''

    classes = [v for v in classnames.values()]

    # images = [AWGN(image, mu=0, sigma=10)]  # numpy array
    start = datetime.datetime.now()

    length_images = 10000

    for images in batch_loop(im_processing_path_list, image_batch):  # Start image processing loop

        images_b = []  # batch of images
        annotations = []  # true labels list
        new_images_list = []

        for img_key in images:  # Load annotation batch
            try:
                annotations.append(dataset[img_key])
                new_images_list.append(img_key)
            except:
                print(f"Annotation for {img_key} not found")

        for i, img_path in enumerate(new_images_list):  # Load images batch
            try:
                image_loaded = cv2.imread(os.path.join(image_path, img_path))
                if noised:
                    image_noised = ASCN.ASCN(image_loaded, 10, 0.4)
                    metr = metric(image_loaded, image_noised)
                    result_file.write(f"{img_path}:  PSNR {metr[1]}    FSIM {metr[2]}   PSNR-HVS-M{metr[3]}")
                if filtered:
                    image_noised = img = DCT.dct_filter(image_noised, 100)

                    result_file.write(
                        f"     PSNR {psnr(image_loaded, image_noised)}    FSIM {fsim(image_loaded, image_noised)}")

                images_b.append(image_loaded)

            except:
                del annotations[i]

        results = model.predict(images_b)
        # results.show()
        for i, (res, labels) in enumerate(zip(results.pred, annotations)):
            # i  # number of image
            # res  # array of bbox [x1, y1, x2, y2, conf, class]
            # labels[class_id]

            t_b[str(i_n)] = []
            p_b[str(i_n)] = []
            for [x1, y1, x2, y2, conf, class_id] in res:
                ious = []
                for lab in labels:
                    ious.append(__calc_iou(lab['bbox'], [x1, y1, x2, y2]))

                iou_id = np.argmax(ious)
                if ious[iou_id] > 0.1:
                    y_t.append(classes.index(list(labels[iou_id].values())[1]))
                    y_p.append(class_id)
                    t_b[str(i_n)].append([list(labels[iou_id].values())[0]])
                    p_b[str(i_n)].append([x1, y1, x2, y2])
            i_n += 1

        length_images -= len(images_b)
        print(f"images left {length_images}")
        if length_images <= 0:
            break

    print(datetime.datetime.now() - start)


    y_t = [int(i) for i in y_t]
    y_p = [int(i) for i in y_p]
    iou = IOU.IoU(t_b, p_b)

    if os.path.exists('processed.json'):
        with open('processed.json', 'rt') as f:
            res_dict = json.load(f)

    res_dict[set_name] = {"y_t": y_t, "y_p": y_p, "IOU": iou}

    with open('processed.json', 'w') as f:
        json.dump(res_dict, f)

# conf = ConfusionMatrix.ConfusionMatrix(y_t, y_p, name=set_name, class_length=None)
#
# iou = IOU.IoU(t_b, p_b)
# print(iou)
# drawMatrix.ConfusionMatrix(conf, save=True, name=f"{set_name} {iou}")
# result_file.write(f"{set_name} IOU: {iou}\n")
#
# result_file.close()
