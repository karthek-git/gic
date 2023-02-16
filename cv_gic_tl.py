"""Train GIC model."""

import os

import tensorflow as tf

import helpers

IMG_SIZE = (224, 224)


def getds():
    """Load dataset."""
    ds_path = "gic_dataset"
    train_dir = os.path.join(ds_path, "train")
    batch_size = 32
    with open("gic_labels.txt") as f:
        gic_labels = f.readlines()
    gic_labels = list(map(lambda x: x.strip(), gic_labels))
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        class_names=gic_labels,
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=1,
        validation_split=0.2,
        subset="both")
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return (train_ds, val_ds)


def getmodel() -> tuple:
    """Compile the model and return."""
    data_augmentation = tf.keras.Sequential(layers=[
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(factor=0.2),
        tf.keras.layers.RandomWidth(factor=0.2),
        tf.keras.layers.RandomHeight(factor=0.2),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ])
    input_shape = IMG_SIZE + (3, )
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False, input_shape=input_shape, pooling="avg")
    base_model.trainable = False
    prediction_layer = tf.keras.layers.Dense(
        units=50, activation=tf.keras.activations.softmax)
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    ouputs = prediction_layer(x)
    model = tf.keras.Model(inputs, ouputs)

    return model, base_model


def train_feature_extractor(model: tf.keras.Model, train_ds: tf.data.Dataset,
                            val_ds: tf.data.Dataset, epochs: int, callbacks):
    """Feature extraction."""
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.2),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=("acc", ))

    history = model.fit(train_ds,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=val_ds)

    return history


def fine_tune(model: tf.keras.Model, base_model: tf.keras.Model,
              train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int,
              initial_epochs: int, history, callbacks):
    """Fine tune the model."""
    base_model.trainable = True
    fine_tune_from = 252

    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=("acc", ))
    total_epochs = initial_epochs + epochs
    history_ft = model.fit(train_ds,
                           epochs=total_epochs,
                           callbacks=callbacks,
                           validation_data=val_ds,
                           initial_epoch=history.epoch[-1])

    return history_ft


def train_model() -> tf.keras.Model:
    """Train model by tl."""
    train_ds, val_ds = getds()
    model, base_model = getmodel()
    initial_epochs = 10
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="artifacts/models/", save_best_only=True)
    cb_earlystop = tf.keras.callbacks.EarlyStopping(patience=5,
                                                    restore_best_weights=True)
    cb_tboard = tf.keras.callbacks.TensorBoard(
        log_dir=helpers.get_tboard_logdir())
    callbacks = (cb_checkpoint, cb_earlystop, cb_tboard,
                 helpers.OverFitMonCB())

    history = train_feature_extractor(model, train_ds, val_ds, initial_epochs,
                                      callbacks)
    history_ft = fine_tune(model,
                           base_model,
                           train_ds,
                           val_ds,
                           epochs=10,
                           initial_epochs=initial_epochs,
                           history=history,
                           callbacks=callbacks)
    helpers.tflite_convert(model, train_ds)
    helpers.plot_metrics(history, history_ft, initial_epochs)

    return model


def main():
    """Train model and exit."""
    train_model()


if __name__ == "__main__":
    main()
