import json
import pathlib
from random import shuffle

import tensorflow as tf

import common


DATA_DIR = pathlib.Path('/home/pablo/un_buen_gesto/hagrid-512p-selected')
ANNOTATIONS_DIR = pathlib.Path('/home/pablo/un_buen_gesto/hagrid-512p-annotations')
BATCH_SIZE = 8

ANNOTATIONS = None

def get_class_id(class_name):
    return common.CLASS_IDS[class_name]


def generate_annotation_lookup():
    global ANNOTATIONS
    ANNOTATIONS = {}
    annotation_filenames = list(ANNOTATIONS_DIR.glob('*/*.json'))
    assert(len(annotation_filenames) > 0)
    for annotation_filename in annotation_filenames:
        if annotation_filename.stem not in common.CLASS_NAMES:
            continue

        with open(annotation_filename) as f:
            annotation = json.load(f)
            for k, v in annotation.items():
                ANNOTATIONS[k] = (v['bboxes'], [get_class_id(l) for l in v['labels']])

generate_annotation_lookup()

def get_boxes_and_classes(stem):
    return ANNOTATIONS[stem]


def get_dataset():
    # gather imamge files
    # image_path_list = list(DATA_DIR.glob('*/*.jpg'))
    image_path_list = list(DATA_DIR.glob('*/*.jpg'))
    image_count = len(image_path_list)
    print(f'got {image_count} images')
    shuffle(image_path_list)

    # gather bounding boxes and labels in lists parallel to `image_list`
    image_str_list = []
    boxes_list = []
    classes_list = []
    for image_path in image_path_list:
        image_str_list.append(str(image_path))
        boxes, classes = get_boxes_and_classes(image_path.stem)
        boxes_list.append(boxes)
        classes_list.append(classes)

    # bundle into a Dataset
    ds = tf.data.Dataset.from_tensor_slices((
        tf.ragged.constant(image_str_list),
        tf.ragged.constant(classes_list),
        tf.ragged.constant(boxes_list)
    ))

    return ds
