import matplotlib
import matplotlib.pyplot as plt
import keras_cv
import keras
import tensorflow as tf

from common import IMAGE_SIZE, NUM_CLASSES, CLASS_MAPPINGS, BOUNDING_BOX_FORMAT
from dataset import get_dataset


AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.004
GLOBAL_CLIPNORM = 10.0
EPOCHS = 5



def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=CLASS_MAPPINGS,
    )
    plt.show()


def parse_image(image_filename):
    # Convert the compressed string to a 3D uint8 tensor
    image = tf.io.read_file(image_filename)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image


resizer = keras_cv.layers.JitteredResize(
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    scale_factor=(1.0, 1.0),
    bounding_box_format=BOUNDING_BOX_FORMAT,
)


augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format=BOUNDING_BOX_FORMAT),
        keras_cv.layers.JitteredResize(
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            scale_factor=(1.0, 1.0),
            bounding_box_format=BOUNDING_BOX_FORMAT,
        ),
    ]
)


def build_yolov8_model():
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_s_backbone_coco",
        load_weights=True,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        global_clipnorm=GLOBAL_CLIPNORM
    )

    model = keras_cv.models.YOLOV8Detector(
        num_classes=NUM_CLASSES,
        bounding_box_format=BOUNDING_BOX_FORMAT,
        backbone=backbone,
        fpn_depth=1,
    )

    model.compile(
        optimizer=optimizer,
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        jit_compile=False
    )

    model.summary()

    return model


def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


def visualize_detections(model, dataset, bounding_box_format):
    for i in range(5):
        images, y_true = next(iter(dataset.take(i+1)))
        y_pred = model.predict(images)
        keras_cv.visualization.plot_bounding_box_gallery(
            images,
            value_range=(0, 255),
            bounding_box_format=bounding_box_format,
            y_pred=y_pred,
            scale=4,
            rows=2,
            cols=2,
            show=True,
            font_scale=0.5,
            class_mapping=CLASS_MAPPINGS,
        )
        plt.show()


def load_sample(image_filename, classes, boxes):
    image = parse_image(image_filename)

    return {
        "images": image,
        "bounding_boxes": {
            "classes": tf.cast(classes, dtype=tf.float32),
            "boxes": boxes,
        }
    }


resizer = keras_cv.layers.JitteredResize(
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    scale_factor=(1.0, 1.0),
    bounding_box_format=BOUNDING_BOX_FORMAT,
)


def main():
    ds = get_dataset()

    # Split the dataset into train and validation sets
    num_val = 20000
    train_ds = ds.skip(num_val)
    val_ds = ds.take(num_val)

    # map image loading function, tweak
    train_ds = train_ds.map(load_sample, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.map(resizer, num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = val_ds.map(load_sample, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    val_ds = val_ds.map(resizer, num_parallel_calls=tf.data.AUTOTUNE)

    # check it out
    visualize_dataset(
        train_ds, bounding_box_format=BOUNDING_BOX_FORMAT, value_range=(0, 255), rows=2, cols=2
    )

    # back to tuples, get ready to rock
    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_yolov8_model()
    print(model.output_names)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='.logs', histogram_freq=1)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="models/yolov8s/yolov8s-{epoch:02d}-{val_loss:.2f}.keras",
        save_weights_only=False,
        save_best_only=False)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tensorboard_callback,
            model_checkpoint_callback
        ]
    )

    # try it out
    visualize_detections(model, dataset=val_ds, bounding_box_format=BOUNDING_BOX_FORMAT)


if __name__ == "__main__":
    main()

