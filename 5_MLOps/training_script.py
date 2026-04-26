import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import mlflow
import mlflow.keras

# =========================
# GPU setup (1 -> 4 logical)
# =========================
physical_gpus = tf.config.list_physical_devices('GPU')

if physical_gpus:
    try:
        tf.config.set_logical_device_configuration(
            physical_gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2048) for _ in range(4)]
        )
    except:
        pass

logical_gpus = tf.config.list_logical_devices('GPU')

print("Physical GPUs:", len(physical_gpus))
print("Logical GPUs:", len(logical_gpus))

# =========================
# Distribution strategy
# =========================
strategy = tf.distribute.MirroredStrategy()
NUM_REPLICAS = strategy.num_replicas_in_sync

print("# of replicas:", NUM_REPLICAS)

# =========================
# Paths (from slurm script)
# =========================
DATA_DIR = os.environ.get("LOCAL_DATA_DIR")

if DATA_DIR is None:
    DATA_DIR = os.environ.get("FLOWER_DATA_DIR")

if DATA_DIR is None:
    DATA_DIR = "/scratch/project_2018566/data/flower_photos"

print("DATA_DIR:", DATA_DIR)

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

# =========================
# Config
# =========================
IMG_SIZE = (150, 150)
PER_REPLICA_BATCH_SIZE = 16
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * NUM_REPLICAS

print("global batch size:", GLOBAL_BATCH_SIZE)

# =========================
# Dataset
# =========================
train_data = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=GLOBAL_BATCH_SIZE
)

val_data = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=GLOBAL_BATCH_SIZE
)

class_names = train_data.class_names
NUM_CLASSES = len(class_names)

print("training data count:", train_data.cardinality().numpy() * GLOBAL_BATCH_SIZE)
print("validation data count:", val_data.cardinality().numpy() * GLOBAL_BATCH_SIZE)

# =========================
# Model
# =========================
with strategy.scope():
    model = keras.Sequential([
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()

# =========================
# MLflow setup
# =========================
jobid = os.environ.get("SLURM_JOB_ID", "local")

mlflow.set_tracking_uri("/scratch/project_2018566/mlruns")
mlflow.set_experiment("flowers")

# =========================
# Training
# =========================
epochs = 20

with mlflow.start_run(run_name=f"NN+{jobid}"):

    start_time = time.time()

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=2
    )

    training_time = time.time() - start_time

    # =========================
    # Log metrics per epoch
    # =========================
    for i in range(len(history.history["loss"])):
        mlflow.log_metric("loss", history.history["loss"][i], step=i)
        mlflow.log_metric("val_loss", history.history["val_loss"][i], step=i)
        mlflow.log_metric("accuracy", history.history["accuracy"][i], step=i)
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][i], step=i)

    # =========================
    # Best epoch
    # =========================
    best_epoch = int(np.argmax(history.history["val_accuracy"]))

    training_accuracy = float(history.history["accuracy"][best_epoch])
    validation_accuracy = float(history.history["val_accuracy"][best_epoch])

    # =========================
    # Log final metrics
    # =========================
    mlflow.log_metric("best_training_accuracy", training_accuracy)
    mlflow.log_metric("best_validation_accuracy", validation_accuracy)

    # =========================
    # Save model
    # =========================
    model_path = "/projappl/project_2018566/models/flower_model_NN.keras"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # log model to MLflow
    mlflow.keras.log_model(model, "model")

# =========================
# Final prints (for grading)
# =========================
print("training_accuracy:", training_accuracy)
print("validation_accuracy:", validation_accuracy)
print("run_time:", training_time)