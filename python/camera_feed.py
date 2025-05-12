import os
import sys
import pathlib
import time

import cv2
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter


from common import IMAGE_SIZE, BOUNDING_BOX_FORMAT, CLASS_MAPPINGS, IS_HEADLESS
from playback import mpd_client
from gesture_recognition import GestureRecognizer



def handle_class(class_name):

    if class_name == "mute":
        mpd_client.stop()
        print("stop")

    elif class_name == "fist":
        mpd_client.play()
        print("play")

def run(camera, recognizer):

    # setup profiling
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    while True:
        t1 = cv2.getTickCount()

        # capture frame
        ret, frame = camera.read()

        if not ret:
            print("Failed to grab frame")
            break

        class_id, class_name = recognizer.recognize(frame)

        text = 'FPS: {0:.2f}'.format(frame_rate_calc)
        if (class_id != -1):
            text += f' detected {class_name}'
            handle_class(class_name)

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
        if not IS_HEADLESS and cv2.waitKey(1) & 0xFF == ord('q'):
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
    recognizer = GestureRecognizer(str(model_file))

    # Open the default webcam (usually the first camera)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Could not open video device")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # run loop
    run(camera, recognizer)


if __name__ == "__main__":
    main()
