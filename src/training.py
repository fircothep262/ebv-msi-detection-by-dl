from typing import Tuple, List

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization

sys.path.append('.')
from src.common import get_dict_labels


img_size = 224  # Width of the patch [pixel].
batch_size = 256

tensorboard_log_dir = 'logs'
cppath = 'checkpoints'

preprocess_dict = {'VGG16': tf.keras.applications.vgg16.preprocess_input,
        'VGG19': tf.keras.applications.vgg19.preprocess_input,
        'ResNet50': tf.keras.applications.resnet50.preprocess_input,
        'EfficientNet': tf.keras.applications.efficientnet.preprocess_input}


def create_dataset(base_model: str, training_data_dir: str, training_groups: List[int],
        validation_data_dir: str, validation_group: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    base_model: 'VGG16', 'VGG19', 'ResNet50', 'EfficientNet'
    Returns: (training dataset, validation dataset)
    If validation_group == -1, validation dataset will be None.
    """
    preprocess = preprocess_dict[base_model]

    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
    }

    def parse_function(example):
        features = tf.io.parse_single_example(example, feature_description)
        label = features['label']

        # float32, [0.0, 255.0]
        image = tf.cast(tf.io.decode_image(features['image'], channels=3), tf.float32)

        image = preprocess(image)
        image.set_shape([img_size, img_size, 3])
        return image, label

    def map_func(x):
        """
        Receive a list of tfrecord files in a specific group, and returns a dataset
        that outputs example in order.
        """
        ds = tf.data.Dataset.from_tensor_slices(x)

        # buffer_size: 100MB
        ds = ds.flat_map(lambda y: tf.data.TFRecordDataset(y, buffer_size=104857600))

        return ds

    file_list = []  # Contains lists of tfrecord files
    for g in training_groups:
        subdir = training_data_dir + fr'\Group{g}'
        files = os.listdir(subdir)
        file_list.append([subdir + '\\' + file for file in files])

    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.interleave(map_func, cycle_length=len(training_groups), block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.repeat().batch(batch_size)
    training_dataset = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if validation_group == -1:
        validation_dataset = None
    else:
        ds = tf.data.TFRecordDataset(validation_data_dir + fr'\Group{validation_group}\00.tfrecord')
        ds = ds.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size)
        validation_dataset = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return training_dataset, validation_dataset


def exec_training(classification_pattern: int, base_model: str,
        training_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset,
        id: str) -> Tuple[List[float], List[float]]:
    """
    classification_pattern:
        1:3-class classification(EBV, MSI, other)
        2:binary classification(EBV+MSI, other)
        3:binary classification(EBV, MSI+other)
        4:binary classification(MSI, EBV+other)
    base_model: 'VGG16', 'VGG19', 'ResNet50', 'EfficientNet'
    """
    dict_labels = get_dict_labels(classification_pattern)
    n_labels = len(dict_labels)
    preprocess = preprocess_dict[base_model]
    preprocess(tf.zeros([1, img_size, img_size, 3]))  # initialize keras globals

    if classification_pattern == 1:
        activation = 'softmax'
        units = n_labels
        loss_function = 'sparse_categorical_crossentropy'
    elif classification_pattern in {2, 3, 4}:
        activation = 'sigmoid'
        units = 1
        loss_function = 'binary_crossentropy'

    cbs = []
    cbs.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir + fr'\logs_{id}',
            histogram_freq=1, write_graph=True))
    
    if not os.path.exists(cppath + fr'\checkpoints_{id}'):
        os.makedirs(cppath + fr'\checkpoints_{id}')
    cbs.append(tf.keras.callbacks.ModelCheckpoint(
            cppath + fr'\checkpoints_{id}' + r'\cp-{epoch:04d}.h5',
            save_weights_only=False, save_freq='epoch', verbose=0))

    val_exists = validation_dataset is not None
    val_loss = []
    val_acc = []

    print(f'Start training. id: {id}')

    strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    with strategy.scope():
        if base_model == 'VGG16':
            base = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                    input_shape=(img_size, img_size, 3))
            base.trainable = False  # Train only FC layers first.
            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(rate=0.4)(x)
            x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)
            x = Dropout(rate=0.4)(x)
            x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)
            x = Dropout(rate=0.7)(x)
            x = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    activation=activation)(x)
            model = tf.keras.Model(inputs=base.input, outputs=x)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=loss_function, metrics=['accuracy'])
            history = model.fit(training_dataset, steps_per_epoch=100, epochs=50,
                    validation_data=validation_dataset,
                    validation_steps=(10 if val_exists else None),
                    verbose=1, callbacks=cbs)
            if val_exists:
                val_loss.extend(history.history['val_loss'])
                val_acc.extend(history.history['val_accuracy'])

            base.trainable = True  # Train all layers.
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=2e-5, momentum=0.9),
                          loss=loss_function, metrics=['accuracy'])
            history = model.fit(training_dataset, steps_per_epoch=100, epochs=300,
                    validation_data=validation_dataset,
                    validation_steps=(10 if val_exists else None),
                    verbose=1, callbacks=cbs)
            if val_exists:
                val_loss.extend(history.history['val_loss'])
                val_acc.extend(history.history['val_accuracy'])
        elif base_model == 'VGG19':
            base = tf.keras.applications.VGG19(include_top=False, weights='imagenet',
                    input_shape=(img_size, img_size, 3))
            base.trainable = False  # Train only FC layers first.
            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(rate=0.4)(x)
            x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)
            x = Dropout(rate=0.4)(x)
            x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)
            x = Dropout(rate=0.7)(x)
            x = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    activation=activation)(x)
            model = tf.keras.Model(inputs=base.input, outputs=x)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=loss_function, metrics=['accuracy'])
            history = model.fit(training_dataset, steps_per_epoch=100, epochs=50,
                    validation_data=validation_dataset,
                    validation_steps=(10 if val_exists else None),
                    verbose=1, callbacks=cbs)
            if val_exists:
                val_loss.extend(history.history['val_loss'])
                val_acc.extend(history.history['val_accuracy'])

            base.trainable = True  # Train all layers.
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=2e-5, momentum=0.9),
                          loss=loss_function, metrics=['accuracy'])
            history = model.fit(training_dataset, steps_per_epoch=100, epochs=300,
                    validation_data=validation_dataset,
                    validation_steps=(10 if val_exists else None),
                    verbose=1, callbacks=cbs)
            if val_exists:
                val_loss.extend(history.history['val_loss'])
                val_acc.extend(history.history['val_accuracy'])
        elif base_model == 'ResNet50':
            base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                    input_shape=(img_size, img_size, 3))
            base.trainable = False  # Train only FC layers first.
            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(rate=0.7)(x)
            x = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    activation=activation)(x)
            model = tf.keras.Model(inputs=base.input, outputs=x)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=loss_function, metrics=['accuracy'])
            history = model.fit(training_dataset, steps_per_epoch=100, epochs=20,
                    validation_data=validation_dataset,
                    validation_steps=(10 if val_exists else None),
                    verbose=1, callbacks=cbs)
            if val_exists:
                val_loss.extend(history.history['val_loss'])
                val_acc.extend(history.history['val_accuracy'])

            base.trainable = True  # Train all layers except for BatchNormalization.
            for layer in base.layers:
                if isinstance(layer, BatchNormalization):
                    layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=2e-5, momentum=0.9),
                    loss=loss_function, metrics=['accuracy'])
            history = model.fit(training_dataset, steps_per_epoch=100, epochs=300,
                    validation_data=validation_dataset,
                    validation_steps=(10 if val_exists else None),
                    verbose=1, callbacks=cbs)
            if val_exists:
                val_loss.extend(history.history['val_loss'])
                val_acc.extend(history.history['val_accuracy'])
        elif base_model == 'EfficientNet':
            base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet',
                    input_shape=(img_size, img_size, 3))
            base.trainable = False  # Train only FC layers first.
            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(rate=0.7)(x)
            x = Dense(units, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                    activation=activation)(x)
            model = tf.keras.Model(inputs=base.input, outputs=x)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=loss_function, metrics=['accuracy'])
            history = model.fit(training_dataset, steps_per_epoch=100, epochs=20,
                    validation_data=validation_dataset,
                    validation_steps=(10 if val_exists else None),
                    verbose=1, callbacks=cbs)
            if val_exists:
                val_loss.extend(history.history['val_loss'])
                val_acc.extend(history.history['val_accuracy'])

            base.trainable = True  # Train all layers except for BatchNormalization.
            for layer in base.layers:
                if isinstance(layer, BatchNormalization):
                    layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
                    loss=loss_function, metrics=['accuracy'])
            history = model.fit(training_dataset, steps_per_epoch=100, epochs=300,
                    validation_data=validation_dataset,
                    validation_steps=(10 if val_exists else None),
                    verbose=1, callbacks=cbs)
            if val_exists:
                val_loss.extend(history.history['val_loss'])
                val_acc.extend(history.history['val_accuracy'])
 
        if val_exists:
            loss, acc = model.evaluate(validation_dataset, steps=100)
            print(f'Validation loss: {loss}')
            print(f'Validation accuracy: {acc}')

    return val_loss, val_acc


def cross_validation(classification_pattern: int, base_model: str,
        training_data_dir: str, validation_data_dir: str, id: str):
    """
    classification_pattern:
        1:3-class classification(EBV, MSI, other)
        2:binary classification(EBV+MSI, other)
        3:binary classification(EBV, MSI+other)
        4:binary classification(MSI, EBV+other)
    base_model: 'VGG16', 'VGG19', 'ResNet50', 'EfficientNet'
    """
    loss_hist = []
    acc_hist = []

    training_dataset, validation_dataset = create_dataset(
            base_model, training_data_dir, [2, 3, 4], validation_data_dir, 1)
    val_loss, val_acc = exec_training(classification_pattern, base_model,
            training_dataset, validation_dataset, id + '_0')
    loss_hist.append(val_loss)
    acc_hist.append(val_acc)

    training_dataset, validation_dataset = create_dataset(
            base_model, training_data_dir, [1, 3, 4], validation_data_dir, 2)
    val_loss, val_acc = exec_training(classification_pattern, base_model,
            training_dataset, validation_dataset, id + '_1')
    loss_hist.append(val_loss)
    acc_hist.append(val_acc)

    training_dataset, validation_dataset = create_dataset(
            base_model, training_data_dir, [1, 2, 4], validation_data_dir, 3)
    val_loss, val_acc = exec_training(classification_pattern, base_model,
            training_dataset, validation_dataset, id + '_2')
    loss_hist.append(val_loss)
    acc_hist.append(val_acc)

    training_dataset, validation_dataset = create_dataset(
            base_model, training_data_dir, [1, 2, 3], validation_data_dir, 4)
    val_loss, val_acc = exec_training(classification_pattern, base_model,
            training_dataset, validation_dataset, id + '_3')
    loss_hist.append(val_loss)
    acc_hist.append(val_acc)

    # Calculate the average of four training instance.
    epochs = len(loss_hist[0])
    loss_hist = [np.mean([x[i] for x in loss_hist]) for i in range(epochs)]
    acc_hist = [np.mean([x[i] for x in acc_hist]) for i in range(epochs)]

    print('======== Final Result ========')
    print(f'loss min: {min(loss_hist)}')
    print(f'loss min acc: {acc_hist[loss_hist.index(min(loss_hist))]}')
    print(f'loss min epoch: {loss_hist.index(min(loss_hist)) + 1}')

    plt.figure(figsize=(8, 8), dpi=100)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(loss_hist) + 1), loss_hist)
    plt.title('loss')
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(acc_hist) + 1), acc_hist)
    plt.title('acc')
    plt.show()
