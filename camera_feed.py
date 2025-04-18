import os
import sys
import pathlib
import time

import cv2
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from mpd import MPDClient


from common import IMAGE_SIZE, BOUNDING_BOX_FORMAT, CLASS_MAPPINGS, IS_HEADLESS


CONFIDENCE_THRESHOLD = 0.8


mpd_client = None


def handle_class(detected_class_id):
    global mpd_client
    if CLASS_MAPPINGS[detected_class_id] == "mute":
        mpd_client.stop()
        print("stop")
    elif CLASS_MAPPINGS[detected_class_id] == "fist":
        mpd_client.play()
        print("play")


def get_detected_class(scores_tensor):
    class_ids = np.argmax(scores_tensor, axis=1)
    class_confs = np.max(scores_tensor, axis=1)

    # get the class with max score
    max_conf = -float("inf")
    detected_class = -1
    for i, conf in enumerate(class_confs):
        if conf > max_conf:
            max_conf = conf
            detected_class = class_ids[i]
            # print(f'current winner {i}, conf {conf}')

    return detected_class if max_conf > CONFIDENCE_THRESHOLD else -1


def prepare_image_for_inference_via_jpeg_codec(image):
    image = tf.io.encode_jpeg(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    image = tf.expand_dims(image, axis=0)

    return image


def prepare_image_for_inference(image):
    frame_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to the expected 640x640
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
    frame_tensor = tf.image.convert_image_dtype(frame_tensor, dtype=tf.float32)  # Normalize to [0.0, 1.0]

    frame_tensor = tf.image.resize_with_pad(frame_tensor, IMAGE_SIZE, IMAGE_SIZE)
    frame_tensor = tf.expand_dims(frame_tensor, axis=0)

    return frame_tensor


def run(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Open the default webcam (usually the first camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # Set video frame width and height (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # setup profiling
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    while True:
        t1 = cv2.getTickCount()

        # capture frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Perform the inference
        input_data = prepare_image_for_inference_via_jpeg_codec(frame)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        scores_tensor = interpreter.get_tensor(output_details[1]['index'])[0]
        detected_class_id = get_detected_class(scores_tensor)

        text = 'FPS: {0:.2f}'.format(frame_rate_calc)
        if (detected_class_id != -1):
            text += f' detected {CLASS_MAPPINGS[detected_class_id]}'
            handle_class(detected_class_id)

        if not IS_HEADLESS:
            # Draw framerate in corner of frame
            cv2.putText(
                frame,
                text,
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2,
                cv2.LINE_AA)
            cv2.imshow('Gesture detector', frame)


        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    # handle cmdline args
    model_file = pathlib.Path(sys.argv[1])
    assert(model_file.suffix == ".tflite")

    # init object detector
    interpreter = Interpreter(model_path=str(model_file))
    interpreter.allocate_tensors()

    # init MPD client
    global mpd_client
    mpd_client = MPDClient()
    mpd_client.connect("localhost", 6600)

    # breakpoint()

    # run loop
    run(interpreter)


if __name__ == "__main__":
    main()
