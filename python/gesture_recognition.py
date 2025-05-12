import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

from common import IMAGE_SIZE, CLASS_MAPPINGS, IS_HEADLESS

CONFIDENCE_THRESHOLD = 0.8


class GestureRecognizer:
    
    def get_detected_class(self, scores_tensor):
        class_ids = np.argmax(scores_tensor, axis=1)
        class_confs = np.max(scores_tensor, axis=1)

        # get the class with max score
        max_conf = -float("inf")
        detected_class = -1
        for i, conf in enumerate(class_confs):
            if conf > max_conf:
                max_conf = conf
                detected_class = class_ids[i]

        return detected_class if max_conf > CONFIDENCE_THRESHOLD else -1


    def prepare_image_for_inference_via_jpeg_codec(self, image):
        image = tf.io.encode_jpeg(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
        image = tf.expand_dims(image, axis=0)

        return image


    def prepare_image_for_inference(self, image):
        frame_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to the expected 640x640
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
        frame_tensor = tf.image.convert_image_dtype(frame_tensor, dtype=tf.float32)  # Normalize to [0.0, 1.0]

        frame_tensor = tf.image.resize_with_pad(frame_tensor, IMAGE_SIZE, IMAGE_SIZE)
        frame_tensor = tf.expand_dims(frame_tensor, axis=0)

        return frame_tensor


    def __init__(self, model_file):
        self.interpreter = Interpreter(model_file)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def recognize(self, frame):
        input_data = self.prepare_image_for_inference_via_jpeg_codec(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        scores_tensor = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        detected_class_id = self.get_detected_class(scores_tensor)
        
        if detected_class_id == -1:
            return -1, ''
        else:
            return detected_class_id, CLASS_MAPPINGS[detected_class_id]

