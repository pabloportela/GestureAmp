import sys

import tensorflow as tf
import keras
import pathlib


assert(len(sys.argv) == 2)
input_file = pathlib.Path(sys.argv[1])
assert(input_file.suffix == ".keras")


# load keras model
model = keras.saving.load_model(str(input_file))

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

print(f"loaded {input_file}")

converter.input_shape = (None, 640, 640, 3)

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.optimizations = []
# converter.target_spec.supported_ops = []

tflite_model = converter.convert()

# save as tflite
output_file = input_file.with_suffix(".tflite")
output_file.write_bytes(tflite_model)
print(f"saved {output_file}")

# quantize
# tflite_quant_model = converter.convert()
# tflite_model_quant_file = pathlib.Path("models/yolov8_s_6_classes/gesture_yolov8_s-08-0.68-quant.tflite")
# tflite_model_quant_file.write_bytes(tflite_quant_model)
