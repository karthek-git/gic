"""Helper functions."""

import os
import pathlib
import time

import matplotlib.pyplot as plt
import tensorflow as tf


def peek_ds(dataset: tf.data.Dataset):
    """Show a few images from dataset."""
    plt.figure(figsize=(10, 10))
    c_names = dataset.class_names

    for images, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(9, 9, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(c_names[labels[i]])
            plt.axis("off")


def plot_metrics(history, history_ft, initial_epochs=None):
    """Plot the metrics."""
    acc = history.history["acc"] + history_ft.history["acc"]
    val_acc = history.history["val_acc"] + history_ft.history["val_acc"]

    loss = history.history["loss"] + history_ft.history["loss"]
    val_loss = history.history["val_loss"] + history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])

    if initial_epochs is not None:
        plt.plot((initial_epochs - 1, initial_epochs - 1),
                 plt.ylim(),
                 label="Start Fine Tuning")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, 1.0])

    if initial_epochs is not None:
        plt.plot((initial_epochs - 1, initial_epochs - 1),
                 plt.ylim(),
                 label="Start Fine Tuning")
    plt.title("Training and Validation Accuracy")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()


class OverFitMonCB(tf.keras.callbacks.Callback):
    """Monitor Overfitting."""

    def on_epoch_end(self, epoch, logs):
        """Print the loss ratio."""
        print(f"val_loss/loss: {logs['val_loss']/logs['loss']}")


TBOARD_ROOT_LOGDIR = "artifacts/tboard/"


def get_tboard_logdir():
    """Get unique logdir name for each run."""
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")

    return os.path.join(TBOARD_ROOT_LOGDIR, run_id)


def tflite_convert(model: tf.keras.Model, dataset: tf.data.Dataset):
    """Convert model to quantized tflite model with optimizations."""

    def gen_representative_data():
        for item, _ in dataset.take(100):
            yield [item]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = gen_representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    pathlib.Path("artifacts/models/gic_uint8_v1.tflite").write_bytes(
        tflite_model)

    return tflite_model
