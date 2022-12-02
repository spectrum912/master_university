from CNN_benchmark.statistical import ConfusionMatrix
from CNN_benchmark.draw import drawMatrix
from CNN_benchmark.statistical import IOU

y_t = [1, 3, 4, 5, 6, 7, 8, 9, 2, 1, 4, 5, 6, 3, 2, 1, 2, 5, 6, 1, 2, 6, 4, 6, 4, 5, 6]
y_p = [1, 2, 3, 5, 6, 7, 8, 9, 2, 1, 4, 5, 6, 3, 2, 1, 2, 5, 6, 1, 2, 6, 4, 6, 4, 5, 6]

conf = ConfusionMatrix.ConfusionMatrix(y_t, y_p)

drawMatrix.ConfusionMatrix(conf, save=True)

true_bboxes = {
    "1": [[0, 0, 200, 100], [150, 20, 1000, 200]],
    "2": [[0, 0, 100, 100], [0, 30, 100, 200]]
}

predict_bboxes = {
    "1": [[0, 0, 200, 100], [150, 20, 1000, 200]],
    "2": [[0, 50, 100, 100], [0, 30, 80, 200]]
}

print(IOU.IoU(true_bboxes, predict_bboxes))