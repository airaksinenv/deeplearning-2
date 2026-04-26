#!/usr/bin/env python3
"""
Train flower CNN on Puhti-style environment.

Default paths are intended for the user's own Puhti project:
- source data is copied by Flower_script.sh from /scratch/project_2018566/data/flower_photos
  to node-local disk
- this script reads FLOWER_DATA_DIR/train and FLOWER_DATA_DIR/val
- the model is saved to /projappl/project_2018566/models/flower_model_NN.keras

The public function remains:
    train_flower_model("pathToTrainDataset", "pathToValidationDataset")
"""

import os

# Must be set before TensorFlow import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


PROJECT_ID = os.environ.get("PROJECT_ID", "project_2018566")
DEFAULT_DATA_DIR = os.environ.get(
    "FLOWER_DATA_DIR", f"/scratch/{PROJECT_ID}/data/flower_photos"
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH", f"/projappl/{PROJECT_ID}/models/flower_model_NN.keras"
)
TRAIN_LOG_PATH = os.environ.get("TRAIN_LOG_PATH", "train_log.out")

IMG_SHAPE = (150, 150)
SEED = 123
N_LOGICAL_GPUS = 4
PER_REPLICA_BATCH_SIZE = 16
EPOCHS = int(os.environ.get("EPOCHS", "30"))
LOGICAL_GPU_MEMORY_MB = int(os.environ.get("LOGICAL_GPU_MEMORY_MB", "2048"))


def _count_images(dataset_path):
    """Count common image files recursively."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")
    return sum(
        len(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
        for ext in extensions
    )


def _configure_logical_gpus():
    """
    Use one physical GPU as four logical GPUs.

    This matches the user's available environment: one V100 GPU is reserved,
    and it is split into 4 logical GPUs with TensorFlow.
    """
    physical_gpus = tf.config.list_physical_devices("GPU")

    if physical_gpus:
        try:
            tf.config.set_logical_device_configuration(
                physical_gpus[0],
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=LOGICAL_GPU_MEMORY_MB
                    )
                    for _ in range(N_LOGICAL_GPUS)
                ],
            )
            print(
                f"Configured 1 physical GPU as {N_LOGICAL_GPUS} logical GPUs "
                f"({LOGICAL_GPU_MEMORY_MB} MB each)."
            )
        except (RuntimeError, ValueError) as err:
            print("Logical GPU configuration was skipped:")
            print(err)
    else:
        print("No physical GPU found. Training will run on CPU if necessary.")

    logical_gpus = tf.config.list_logical_devices("GPU")
    print("Physical GPUs:", len(physical_gpus))
    print("Logical GPUs:", len(logical_gpus))
    return logical_gpus


def _build_model(num_classes):
    """Build a CNN from scratch. No transfer learning is used."""
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.08, 0.08),
        ],
        name="data_augmentation",
    )

    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SHAPE[0], IMG_SHAPE[1], 3)),
            data_augmentation,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.30),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def train_flower_model(pathToTrainDataset, pathToValidationDataset):
    """
    Train the flower model and save the best Keras model.

    Parameters
    ----------
    pathToTrainDataset : str
        Directory with one subdirectory per class for training data.
    pathToValidationDataset : str
        Directory with one subdirectory per class for validation data.

    Returns
    -------
    None
    """
    with open(TRAIN_LOG_PATH, "w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            tf.get_logger().setLevel("ERROR")
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            print("Training flower model")
            print("train dataset:", pathToTrainDataset)
            print("validation dataset:", pathToValidationDataset)
            print("model path:", MODEL_PATH)
            print("TensorFlow version:", tf.__version__)
            print("Keras version:", keras.__version__)
            print()

            _configure_logical_gpus()
            strategy = tf.distribute.MirroredStrategy()

            num_replicas = strategy.num_replicas_in_sync
            global_batch_size = PER_REPLICA_BATCH_SIZE * num_replicas

            train_count = _count_images(pathToTrainDataset)
            validation_count = _count_images(pathToValidationDataset)

            print("# of replicas:", num_replicas)
            print("global batch size:", global_batch_size)
            print("training data count:", train_count)
            print("validation data count:", validation_count)
            print()

            train_data = keras.utils.image_dataset_from_directory(
                pathToTrainDataset,
                labels="inferred",
                label_mode="int",
                image_size=IMG_SHAPE,
                batch_size=global_batch_size,
                shuffle=True,
                seed=SEED,
            )

            validation_data = keras.utils.image_dataset_from_directory(
                pathToValidationDataset,
                labels="inferred",
                label_mode="int",
                image_size=IMG_SHAPE,
                batch_size=global_batch_size,
                shuffle=False,
            )

            class_names = train_data.class_names
            num_classes = len(class_names)

            print("class names:", class_names)
            print("number of classes:", num_classes)
            print()

            with strategy.scope():
                model = _build_model(num_classes)
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )

            print("Model summary:")
            model.summary(print_fn=print)
            print()

            Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

            checkpoint = keras.callbacks.ModelCheckpoint(
                MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            )

            start_time = time.time()
            history = model.fit(
                train_data,
                validation_data=validation_data,
                epochs=EPOCHS,
                callbacks=[checkpoint],
                verbose=2,
            )
            run_time = time.time() - start_time

            best_epoch_index = int(np.argmax(history.history["val_accuracy"]))
            best_epoch = best_epoch_index + 1
            training_accuracy = float(history.history["accuracy"][best_epoch_index])
            validation_accuracy = float(history.history["val_accuracy"][best_epoch_index])

            print()
            print("Results")
            print("best_epoch:", best_epoch)
            print("# of replicas:", num_replicas)
            print("global batch size:", global_batch_size)
            print("training_accuracy:", training_accuracy)
            print("validation_accuracy:", validation_accuracy)
            print("run_time:", run_time)
            print("saved model:", MODEL_PATH)
            print("Finished training.")

    return None


if __name__ == "__main__":
    train_dir = os.path.join(DEFAULT_DATA_DIR, "train")
    val_dir = os.path.join(DEFAULT_DATA_DIR, "val")
    train_flower_model(train_dir, val_dir)
