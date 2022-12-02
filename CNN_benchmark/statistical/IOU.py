import numpy as np


def __calc_iou(true, predicted):
    if type(true[0]) is list:
        true = true[0]
    xA = max(true[0], predicted[0])
    yA = max(true[1], predicted[1])
    xB = min(true[2], predicted[2])
    yB = min(true[3], predicted[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (true[2] - true[0] + 1) * (true[3] - true[1] + 1)
    boxBArea = (predicted[2] - predicted[0] + 1) * (predicted[3] - predicted[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def IoU(y_true, y_predict, consider_empty=False):
    """
    Calculate min, max and mean IoU for predicted data
    bbox_array = {
        '1': [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax] ...]...
    }
    :param y_true: bbox_array
    :param y_predict: bbox_array
    :param consider_empty: is consider empty bboxes for image
    :return: array of IoU [min, mean, max]
    """
    IoU_array = []
    for image in y_true.keys():
        if image in y_predict:
            for bbox_true in y_true[image]:
                iou_local = []
                for bbox_predict in y_predict[image]:
                    iou_local.append(__calc_iou(bbox_true, bbox_predict))
                iou_local_idx = np.argmax(iou_local)
                IoU_array.append(iou_local[iou_local_idx])
        elif consider_empty:
            IoU_array.extend([0 for _ in y_true[image]])
    return [np.min(IoU_array), np.mean(IoU_array), np.max(IoU_array)]
