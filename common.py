import os
import matplotlib



os.environ["KERAS_BACKEND"] = "tensorflow"

CLASS_NAMES = ['dislike', 'fist', 'mute', 'rock', 'stop', 'no_gesture', 'peace', 'palm']
CLASS_IDS = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))
CLASS_MAPPINGS = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))
NUM_CLASSES = len(CLASS_NAMES)
BOUNDING_BOX_FORMAT = "rel_xywh"
IMAGE_SIZE = 640
IS_HEADLESS = os.environ.get('DISPLAY', '') == ''


if IS_HEADLESS:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
