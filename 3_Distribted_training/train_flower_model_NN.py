#!/usr/bin/env python3
"""
Train a flower image classifier with TensorFlow/Keras.

Required public function:
    train_flower_model("pathToTrainDataset", "pathToValidationDataset")

The function:
- supports multi GPU training with MirroredStrategy
- uses validation data while training
- writes logs to train_log.out
- saves the best Keras model to ./models/flower_model_NN.keras
- returns no value
"""

import os

# Must be set before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import time
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMG_SHAPE = (150, 150)
SEED = 123
N_LOGICAL_GPUS = 4
PER_REPLICA_BATCH_SIZE = 16
EPOCHS = 30
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "flower_model_NN.keras")
TRAIN_LOG_PATH = "train_log.out"


def _count_images(dataset_path):
    """Count common image files recursively."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")
    return sum(
        len(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
        for ext in extensions
    )


def _configure_logical_gpus():
    """
    Configure one physical GPU as four logical GPUs if possible.

    This matches the notebook exercise requirement. If the environment already
    initialized GPUs, TensorFlow may not allow changing logical devices anymore;
    in that case training continues with the visible devices.
    """
    physical_gpus = tf.config.list_physical_devices("GPU")

    if physical_gpus:
        try:
            tf.config.set_logical_device_configuration(
                physical_gpus[0],
                [
                    tf.config.LogicalDeviceConfiguration(memory_limit=2048)
                    for _ in range(N_LOGICAL_GPUS)
                ],
            )
            print(f"Configured 1 physical GPU as {N_LOGICAL_GPUS} logical GPUs.")
        except RuntimeError as err:
            print("Logical GPU configuration was skipped:")
            print(err)
        except ValueError as err:
            print("Logical GPU configuration was skipped:")
            print(err)
    else:
        print("No physical GPU found. Training will run on CPU if necessary.")

    logical_gpus = tf.config.list_logical_devices("GPU")
    print("Physical GPUs:", len(physical_gpus))
    print("Logical GPUs:", len(logical_gpus))
    return logical_gpus


def _build_model(num_classes):
    """Build the flower CNN from scratch (no transfer learning)."""
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
    Train a flower model and save the best model to ./models/flower_model_NN.keras.

    Parameters
    ----------
    pathToTrainDataset : str
        Path to training dataset directory. The directory must contain one
        subdirectory per class.
    pathToValidationDataset : str
        Path to validation dataset directory. The directory must contain one
        subdirectory per class.

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
            print("TensorFlow version:", tf.__version__)
            print("Keras version:", keras.__version__)
            print()

            logical_gpus = _configure_logical_gpus()
            strategy = tf.distribute.MirroredStrategy()

            num_replicas = strategy.num_replicas_in_sync
            global_batch_size = PER_REPLICA_BATCH_SIZE * num_replicas

            train_count = _count_images(pathToTrainDataset)
            validation_count = _count_images(pathToValidationDataset)

            print("# replicas:", num_replicas)
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

            os.makedirs(MODEL_DIR, exist_ok=True)

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

            training_time = time.time() - start_time

            best_epoch_index = int(np.argmax(history.history["val_accuracy"]))
            best_epoch = best_epoch_index + 1
            training_accuracy = float(history.history["accuracy"][best_epoch_index])
            validation_accuracy = float(history.history["val_accuracy"][best_epoch_index])

            # Load the best saved model and evaluate it once more for the log.
            best_model = keras.models.load_model(MODEL_PATH)
            train_loss_eval, train_accuracy_eval = best_model.evaluate(train_data, verbose=0)
            val_loss_eval, val_accuracy_eval = best_model.evaluate(validation_data, verbose=0)

            print()
            print("Results")
            print("best_epoch:", best_epoch)
            print("training_time:", training_time)
            print("training_accuracy from history:", training_accuracy)
            print("validation_accuracy from history:", validation_accuracy)
            print("training_accuracy evaluated best model:", float(train_accuracy_eval))
            print("validation_accuracy evaluated best model:", float(val_accuracy_eval))
            print("training_loss evaluated best model:", float(train_loss_eval))
            print("validation_loss evaluated best model:", float(val_loss_eval))
            print("saved model:", MODEL_PATH)
            print("Finished training.")

    return None


if __name__ == "__main__":
    base_dir = "/scratch/project_2018566/data/flower_photos"

    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    train_flower_model(train_dir, val_dir)
