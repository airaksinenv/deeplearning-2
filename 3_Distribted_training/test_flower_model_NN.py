#!/usr/bin/env python3
"""
Test a trained flower image classifier with TensorFlow/Keras.

Required public function:
    test_flower_model("pathToTestDataset")

The function:
- loads ./models/flower_model_NN.keras
- writes logs to test_log.out
- logs UTC time, run time, tested images, correct %, test loss per image
- returns tested images, correct %, test loss per image
"""

import os

# Must be set before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import time
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone

import tensorflow as tf
from tensorflow import keras


IMG_SHAPE = (150, 150)
BATCH_SIZE = 64
MODEL_PATH = "./models/flower_model_NN.keras"
TEST_LOG_PATH = "test_log.out"


def _count_images(dataset_path):
    """Count common image files recursively."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")
    return sum(
        len(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
        for ext in extensions
    )


def test_flower_model(pathToTestDataset):
    """
    Load the saved model and evaluate it with test data.

    Parameters
    ----------
    pathToTestDataset : str
        Path to test dataset directory. The directory must contain one
        subdirectory per class.

    Returns
    -------
    tuple
        (tested images, correct %, test loss per image)
    """
    tested_images = _count_images(pathToTestDataset)
    correct_percent = 0.0
    test_loss_per_image = 0.0

    with open(TEST_LOG_PATH, "w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            tf.get_logger().setLevel("ERROR")

            utc_time = datetime.now(timezone.utc).isoformat()
            print("Testing flower model")
            print("UTC time:", utc_time)
            print("test dataset:", pathToTestDataset)
            print("model path:", MODEL_PATH)
            print("tested images:", tested_images)
            print()

            start_time = time.time()

            model = keras.models.load_model(MODEL_PATH)

            test_data = keras.utils.image_dataset_from_directory(
                pathToTestDataset,
                labels="inferred",
                label_mode="int",
                image_size=IMG_SHAPE,
                batch_size=BATCH_SIZE,
                shuffle=False,
            )

            test_loss_per_image, test_accuracy = model.evaluate(test_data, verbose=2)
            correct_percent = float(test_accuracy * 100.0)
            run_time = time.time() - start_time

            print()
            print("Results")
            print("UTC time:", utc_time)
            print("run time:", run_time)
            print("tested images:", tested_images)
            print("correct %:", correct_percent)
            print("test loss per image:", float(test_loss_per_image))
            print("Finished testing.")

    return tested_images, correct_percent, float(test_loss_per_image)


if __name__ == "__main__":
    base_dir = "/scratch/project_2018566/data/flower_photos"

    test_dir = os.path.join(base_dir, "val")

    test_flower_model(test_dir)
