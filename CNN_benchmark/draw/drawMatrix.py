import matplotlib.pyplot as plt
import numpy as np
from . import utils

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']


def ConfusionMatrix(confusion_matrix, labels=None, show=True, save=False, name="conflusion_matrix"):
    if labels is None:
        labels = names  # np.arange(np.shape(confusion_matrix)[0])

    fig, ax = plt.subplots(figsize=(32, 32))
    ax.imshow(confusion_matrix)

    for i in range(np.shape(confusion_matrix)[0]):
        for j in range(np.shape(confusion_matrix)[1]):
            text = ax.text(j, i, f"{int(confusion_matrix[i, j])}",
                           ha="center", va="center", color="w")

    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.xticks(range(np.shape(confusion_matrix)[0]), labels[:np.shape(confusion_matrix)[0]], rotation='vertical')
    plt.yticks(range(np.shape(confusion_matrix)[0]), labels[:np.shape(confusion_matrix)[0]])

    if save:
        utils.__result_dir_create("results")
        plt.savefig(f'results/{name}.jpg')
    if show:
        plt.show()
