import sys
import pathlib

import numpy as np
import cv2
import tensorflow as tf
import keras
import keras_cv

from common import IMAGE_SIZE, BOUNDING_BOX_FORMAT, CLASS_MAPPINGS



# parse args

def parse_args():
    assert(len(sys.argv) == 3)
    model_file = pathlib.Path(sys.argv[1])
    assert(model_file.suffix == ".keras" or model_file.suffix == ".tflite")

    image_file = pathlib.Path(sys.argv[2])
    assert(image_file.suffix == ".jpg" or image_file.suffix == ".jpeg" or image_file.suffix == ".png")

    return model_file, image_file


def infer_tflite(model_file, image_file):

    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # image = get_image_as_tensor(image_file)

    image = cv2.imread(image_file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    input_data = np.expand_dims(image_resized / 255.0, axis=0).astype(np.float32)

    breakpoint()
    print("Input data shape:", input_data.shape)
    print("Expected shape:", input_details[0]['shape'])

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    detections = output_data[0]  # remove batch dimension

    for det in detections:
        breakpoint()
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.3:
            print(f"Class: {int(cls)}, Confidence: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")


    print("Input shape:", input_details[0]['shape'])  # Usually (1, 640, 640, 3)


def get_image_as_tensor(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    image = tf.expand_dims(image, axis=0)

    return image


def infer_keras(model_file, image_file):

    model = keras.saving.load_model(model_file)
    image = get_image_as_tensor(image_file)

    # infer
    y_pred = model.predict(image)
    print(y_pred)

    # post-process
    keras_cv.visualization.plot_bounding_box_gallery(
        image,
        value_range=(0, 255),
        bounding_box_format=BOUNDING_BOX_FORMAT,
        y_pred=y_pred,
        scale=1,
        rows=1,
        cols=1,
        show=True,
        font_scale=0.5,
        class_mapping=CLASS_MAPPINGS
    )


def main():
    model_file, image_file = parse_args()

    if (model_file.suffix == ".keras"):
        infer_keras(str(model_file), str(image_file))

    elif (model_file.suffix == ".tflite"):
        infer_tflite(str(model_file), str(image_file))


main()

