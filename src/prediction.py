from typing import List

import os
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append('.')
from src.common import assign_label


tma_list_file = r'data\tma_wsi_data.xlsx'
tma_sheet_name = 'list'
tcga_list_file = r'data\tcga_wsi_data.xlsx'
tcga_sheet_name = 'list'

preprocess_dict = {'VGG16': tf.keras.applications.vgg16.preprocess_input,
        'VGG19': tf.keras.applications.vgg19.preprocess_input,
        'ResNet50': tf.keras.applications.resnet50.preprocess_input,
        'EfficientNet': tf.keras.applications.efficientnet.preprocess_input}

img_size = 224  # Width of the patch [pixel].
batch_size = 256  # The number of patches used to make prediction for a patient.


def tma_patient_level_prediction(classification_pattern: int, base_model: str, model_file: str,
        tma_test_image_dir: str, test_group: int):
    """
    classification_pattern:
        1:3-class classification(EBV, MSI, other)
        2:binary classification(EBV+MSI, other)
        3:binary classification(EBV, MSI+other)
        4:binary classification(MSI, EBV+other)
    base_model: 'VGG16', 'VGG19', 'ResNet50', 'EfficientNet'
    """
    data_types = {'N_TMA': int, 'N_Case': int, 'Molecular3': str, 'Group': int}
    df = pd.read_excel(tma_list_file, tma_sheet_name, dtype=data_types)
    model = tf.keras.models.load_model(model_file)
    np.set_printoptions(formatter={'int': '{:03d}'.format, 'float': '{:.04f}'.format})
    preprocess = preprocess_dict[base_model]
    preprocess(tf.zeros([1, img_size, img_size, 3]))  # initialize keras globals

    for _, row in df.iterrows():
        tma = row['N_TMA']
        case = row['N_Case']
        label_name = assign_label(row['Molecular3'], classification_pattern)
        group = row['Group']

        if (group != test_group):
            continue

        images = []
        image_dir = tma_test_image_dir + fr'\{tma}\{case:0>2}'
        files = os.listdir(image_dir)
        for i in range(batch_size):
            image_file = random.choice(files)
            image = tf.keras.preprocessing.image.load_img(image_dir + '\\' + image_file)
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(preprocess(image))

        prediction = model.predict(np.asarray(images, dtype=np.float32))
        prediction = np.mean(prediction, axis=0)  # Aggregate the result of multiple patches.
        if classification_pattern in {2, 3, 4}:
            # Convert to the probability of target class (EBV+MSI, EBV, MSI, respectively)
            prediction[0] = 1.0 - prediction[0]
        print(f'TMA:{tma}, Case:{case:0>2}, Ground Truth:{label_name:10}, Prediction:{prediction}')


def tcga_patient_level_prediction(classification_pattern: int, base_model: str, model_file: str,
        tcga_test_image_dir: str, test_groups: List[int]):
    """
    classification_pattern:
        1:3-class classification(EBV, MSI, other)
        2:binary classification(EBV+MSI, other)
        3:binary classification(EBV, MSI+other)
        4:binary classification(MSI, EBV+other)
    base_model: 'VGG16', 'VGG19', 'ResNet50', 'EfficientNet'
    """
    data_types = {'TCGA barcode': str, 'Molecular Subtype': str, 'Group': int}
    df = pd.read_excel(tcga_list_file, tcga_sheet_name, dtype=data_types)
    model = tf.keras.models.load_model(model_file)
    np.set_printoptions(formatter={'int': '{:03d}'.format, 'float': '{:.04f}'.format})
    preprocess = preprocess_dict[base_model]
    preprocess(tf.zeros([1, img_size, img_size, 3]))  # initialize keras globals

    for _, row in df.iterrows():
        code = row['TCGA barcode']
        label_name = assign_label(row['Molecular Subtype'], classification_pattern)
        group = row['Group']

        if group not in test_groups:
            continue

        images = []
        image_dir = tcga_test_image_dir + fr'\{code}'
        files = os.listdir(image_dir)
        for i in range(batch_size):
            image_file = random.choice(files)
            image = tf.keras.preprocessing.image.load_img(image_dir + '\\' + image_file)
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(preprocess(image))

        prediction = model.predict(np.asarray(images, dtype=np.float32))
        prediction = np.mean(prediction, axis=0)  # Aggregate the result of multiple patches.
        if classification_pattern in {2, 3, 4}:
            # Convert to the probability of target class (EBV+MSI, EBV, MSI, respectively)
            prediction[0] = 1.0 - prediction[0]
        print(f'Code:{code}, Ground Truth:{label_name:10}, Prediction:{prediction}')
