import os
import json
from CNN_benchmark.draw import drawMatrix
from CNN_benchmark.statistical import ConfusionMatrix

result_file = open("res.txt", 'a')

if os.path.exists('processed.json'):
    with open('processed.json', 'rt') as f:
        res_dict = json.load(f)

for set_name in res_dict.keys():
    y_t = res_dict[set_name]['y_t']
    y_p = res_dict[set_name]['y_p']
    iou = res_dict[set_name]['IOU']

    conf = ConfusionMatrix.ConfusionMatrix(y_t, y_p, name=set_name, class_length=5)
    print(iou)
    drawMatrix.ConfusionMatrix(conf, save=True, show=False, name=f"{set_name} {iou}")
    result_file.write(f"{set_name} IOU: {iou}\n")

    result_file.close()