import json
import numpy as np


def LoadDataset(filename: str):
    data = json.load(open(filename))
    imagelist = {item['id']: item['file_name'] for item in data['images']}
    classes = {item['id']: item['name'] for item in data['categories']}
    annotations = {}
    for item in data['annotations']:
        image_name = imagelist[item['image_id']]
        if image_name not in annotations:
            annotations[image_name] = []
        annotations[image_name].append({'bbox': item['bbox'],
                                        "classname": classes[item['category_id']]})
    return annotations, classes


if __name__ == "__main__":
    dataset, classes = LoadDataset('/home/rostislavts/Documents/CNN_benchmark/dataset/coco/annotations'
                                   '/instances_train2017.json')
    print(len(dataset))
