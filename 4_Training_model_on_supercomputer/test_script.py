#!/usr/bin/env python3
"""
Test a trained flower CNN on Puhti-style environment.

The public function remains:
    test_flower_model("pathToTestDataset")
"""

import os

# Must be set before TensorFlow import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import time
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone

import tensorflow as tf
from tensorflow import keras


PROJECT_ID = os.environ.get("PROJECT_ID", "project_2018566")
DEFAULT_DATA_DIR = os.environ.get(
    "FLOWER_DATA_DIR", f"/scratch/{PROJECT_ID}/data/flower_photos"
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH", f"/projappl/{PROJECT_ID}/models/flower_model_NN.keras"
)
TEST_LOG_PATH = os.environ.get("TEST_LOG_PATH", "test_log.out")

IMG_SHAPE = (150, 150)
BATCH_SIZE = 64


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
        Directory with one subdirectory per class for test data.

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
            print("run_time:", run_time)
            print("tested images:", tested_images)
            print("correct %:", correct_percent)
            print("test loss per image:", float(test_loss_per_image))
            print("Finished testing.")

    return tested_images, correct_percent, float(test_loss_per_image)


if __name__ == "__main__":
    test_dir = os.path.join(DEFAULT_DATA_DIR, "test")
    if not os.path.isdir(test_dir):
        # User's local data currently has train/val; use val as test fallback.
        test_dir = os.path.join(DEFAULT_DATA_DIR, "val")
    test_flower_model(test_dir)
