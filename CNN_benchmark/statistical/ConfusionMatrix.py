import numpy as np

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']


def ConfusionMatrix(y_true, y_predict, name="conf_m", class_length=None):
    file_result = open(f"results/{name}.txt", 'a')

    if class_length is None or (class_length != len(set(y_true)) or class_length != len(set(y_predict))):
        class_length = max(max(set(y_true)), max(set(y_predict))) + 1

    conflusion_matrix = np.zeros((int(class_length), int(class_length)), dtype=np.float32)
    conf_true = np.zeros((int(class_length), int(class_length)), dtype=np.float32)
    table = np.zeros(int(class_length), dtype=np.float32)
    y_true = np.array(y_true)

    y_predict = np.array(y_predict)
    for true_i, predict_i in zip(y_true.flatten(), y_predict.flatten()):
        conflusion_matrix[int(predict_i)][true_i] += 1

    summ_true = 0.0
    summ_predict = 0.0
    for c in y_true.flatten():
        conf_true[c][c] += 1

    for c in range(len(conf_true[0])):
        if conf_true[c][c] != 0:
            table[c] = float(float(conflusion_matrix[c][c]) / float(conf_true[c][c]))
        else:
            table[c] = 0.0
        summ_true += conf_true[c][c]

    for i in range(len(y_true)):
        if y_true[i] == y_predict[i]:
            summ_predict += 1

    table = sorted(table, reverse=True)
    # table = table[:20]
    for c in range(len(table)):
        file_result.write(f"{names[c]} : {table[c]}\n")

    file_result.write(f"total : {summ_predict / summ_true}\n")
    file_result.close()

    return conflusion_matrix
