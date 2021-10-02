import os
import random
import sys

import pandas as pd
import tensorflow as tf

sys.path.append('.')
from src.common import assign_label, get_dict_labels


tma_list_file = r'data\tma_wsi_data.xlsx'
tma_sheet_name = 'list'
tcga_list_file = r'data\tcga_wsi_data.xlsx'
tcga_sheet_name = 'list'

images_num = 100000  # Number of images included in a tfrecord file.


def create_tfrecord(source: str, classification_pattern: int, data_dir: str, output_dir: str,
        output_filename: str, target_group: int) -> None:
    """
    Create tfrecord from patches in a directory.
    source: 'tma' or 'tcga'
    classification_pattern:
        1:3-class classification(EBV, MSI, other)
        2:binary classification(EBV+MSI, other)
        3:binary classification(EBV, MSI+other)
        4:binary classification(MSI, EBV+other)
    """
    dict_labels = get_dict_labels(classification_pattern)
    n_labels = len(dict_labels)

    # Create target file list.
    image_directories = [[] for i in range(n_labels)]
    if source == 'tma':
        data_types = {'N_TMA': int, 'N_Case': int, 'Molecular3': str, 'Group': int}
        df = pd.read_excel(tma_list_file, tma_sheet_name, dtype=data_types)
        for _, row in df.iterrows():
            tma = row['N_TMA']
            case = row['N_Case']
            label_name = assign_label(row['Molecular3'], classification_pattern)
            group = row['Group']
            if group == target_group:
                image_directory = data_dir + fr'\{tma}\{case:0>2}'
                files = os.listdir(image_directory)
                image_directories[dict_labels[label_name]].append((image_directory, files))
    elif source == 'tcga':
        data_types = {'TCGA barcode': str, 'Molecular Subtype': str, 'Group': int}
        df = pd.read_excel(tcga_list_file, tcga_sheet_name, dtype=data_types)
        for _, row in df.iterrows():
            code = row['TCGA barcode']
            label_name = assign_label(row['Molecular Subtype'], classification_pattern)
            group = row['Group']
            if group == target_group:
                image_directory = data_dir + fr'\{code}'
                files = os.listdir(image_directory)
                image_directories[dict_labels[label_name]].append((image_directory, files))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = output_dir + '\\' + output_filename
    print(f'Create TFRecord: {output_path}')
    print('Number of cases:')
    for key, val in dict_labels.items():
        print(f'{key}: {len(image_directories[val])}')

    with tf.io.TFRecordWriter(output_path) as writer:
        # Write randomly selected images to tfrecord file.
        # Total number of images are the same between classes.
        for i in range(images_num):
            label = random.randrange(n_labels)
            cases = image_directories[label]
            case = random.choice(cases)
            file = case[0] + '\\' + random.choice(case[1])
            image = open(file, 'rb').read()
            feature = {
                'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            print(f'\r  Writing: ({i + 1}/{images_num})', end='')
    print()
    print('Completed')


if __name__ == '__main__':
    # Training dataset for "EBV vs MSI vs other" with data augmentation.
    data_dir = r'data\patches\tma_with_aug'
    tcga_data_dir = r'data\patches\tcga_with_aug'
    output_dir = r'data\tfrecords\task1_with_aug'
    for i in range(3):
        create_tfrecord('tma', 1, data_dir, output_dir + r'\Group1', f'{i:0>2}.tfrecord', 1)
        create_tfrecord('tma', 1, data_dir, output_dir + r'\Group2', f'{i:0>2}.tfrecord', 2)
        create_tfrecord('tma', 1, data_dir, output_dir + r'\Group3', f'{i:0>2}.tfrecord', 3)
        create_tfrecord('tma', 1, data_dir, output_dir + r'\Group4', f'{i:0>2}.tfrecord', 4)

        # A part of tcga cases.
        create_tfrecord('tcga', 1, tcga_data_dir, output_dir + r'\Group9', f'{i:0>2}.tfrecord', 0)

    # Training dataset for "EBV vs MSI vs other" without data augmentation (for validation).
    data_dir = r'data\patches\tma_no_aug'
    output_dir = r'data\tfrecords\task1_no_aug'
    for i in range(1):
        create_tfrecord('tma', 1, data_dir, output_dir + r'\Group1', f'{i:0>2}.tfrecord', 1)
        create_tfrecord('tma', 1, data_dir, output_dir + r'\Group2', f'{i:0>2}.tfrecord', 2)
        create_tfrecord('tma', 1, data_dir, output_dir + r'\Group3', f'{i:0>2}.tfrecord', 3)
        create_tfrecord('tma', 1, data_dir, output_dir + r'\Group4', f'{i:0>2}.tfrecord', 4)

    # Training dataset for "EBV+MSI vs other" with data augmentation.
    data_dir = r'data\patches\tma_with_aug'
    tcga_data_dir = r'data\patches\tcga_with_aug'
    output_dir = r'data\tfrecords\task2_with_aug'
    for i in range(3):
        create_tfrecord('tma', 2, data_dir, output_dir + r'\Group1', f'{i:0>2}.tfrecord', 1)
        create_tfrecord('tma', 2, data_dir, output_dir + r'\Group2', f'{i:0>2}.tfrecord', 2)
        create_tfrecord('tma', 2, data_dir, output_dir + r'\Group3', f'{i:0>2}.tfrecord', 3)
        create_tfrecord('tma', 2, data_dir, output_dir + r'\Group4', f'{i:0>2}.tfrecord', 4)

        # A part of tcga cases.
        create_tfrecord('tcga', 2, tcga_data_dir, output_dir + r'\Group9', f'{i:0>2}.tfrecord', 0)

    # Training dataset for "EBV+MSI vs other" without data augmentation.
    data_dir = r'data\patches\tma_no_aug'
    tcga_data_dir = r'data\patches\tcga_no_aug'
    output_dir = r'data\tfrecords\task2_no_aug'
    for i in range(3):
        create_tfrecord('tma', 2, data_dir, output_dir + r'\Group1', f'{i:0>2}.tfrecord', 1)
        create_tfrecord('tma', 2, data_dir, output_dir + r'\Group2', f'{i:0>2}.tfrecord', 2)
        create_tfrecord('tma', 2, data_dir, output_dir + r'\Group3', f'{i:0>2}.tfrecord', 3)
        create_tfrecord('tma', 2, data_dir, output_dir + r'\Group4', f'{i:0>2}.tfrecord', 4)

        # A part of tcga cases.
        create_tfrecord('tcga', 2, tcga_data_dir, output_dir + r'\Group9', f'{i:0>2}.tfrecord', 0)

    # Training dataset for "EBV vs MSI+other" with data augmentation.
    data_dir = r'data\patches\tma_with_aug'
    output_dir = r'data\tfrecords\task3_with_aug'
    for i in range(3):
        create_tfrecord('tma', 3, data_dir, output_dir + r'\Group1', f'{i:0>2}.tfrecord', 1)
        create_tfrecord('tma', 3, data_dir, output_dir + r'\Group2', f'{i:0>2}.tfrecord', 2)
        create_tfrecord('tma', 3, data_dir, output_dir + r'\Group3', f'{i:0>2}.tfrecord', 3)
        create_tfrecord('tma', 3, data_dir, output_dir + r'\Group4', f'{i:0>2}.tfrecord', 4)

    # Training dataset for "EBV vs MSI+other" without data augmentation (for validation).
    data_dir = r'data\patches\tma_no_aug'
    output_dir = r'data\tfrecords\task3_no_aug'
    for i in range(1):
        create_tfrecord('tma', 3, data_dir, output_dir + r'\Group1', f'{i:0>2}.tfrecord', 1)
        create_tfrecord('tma', 3, data_dir, output_dir + r'\Group2', f'{i:0>2}.tfrecord', 2)
        create_tfrecord('tma', 3, data_dir, output_dir + r'\Group3', f'{i:0>2}.tfrecord', 3)
        create_tfrecord('tma', 3, data_dir, output_dir + r'\Group4', f'{i:0>2}.tfrecord', 4)

    # Training dataset for "MSI vs EBV+other" with data augmentation.
    data_dir = r'data\patches\tma_with_aug'
    output_dir = r'data\tfrecords\task4_with_aug'
    for i in range(3):
        create_tfrecord('tma', 4, data_dir, output_dir + r'\Group1', f'{i:0>2}.tfrecord', 1)
        create_tfrecord('tma', 4, data_dir, output_dir + r'\Group2', f'{i:0>2}.tfrecord', 2)
        create_tfrecord('tma', 4, data_dir, output_dir + r'\Group3', f'{i:0>2}.tfrecord', 3)
        create_tfrecord('tma', 4, data_dir, output_dir + r'\Group4', f'{i:0>2}.tfrecord', 4)

    # Training dataset for "MSI vs EBV+other" without data augmentation (for validation).
    data_dir = r'data\patches\tma_no_aug'
    output_dir = r'data\tfrecords\task4_no_aug'
    for i in range(1):
        create_tfrecord('tma', 4, data_dir, output_dir + r'\Group1', f'{i:0>2}.tfrecord', 1)
        create_tfrecord('tma', 4, data_dir, output_dir + r'\Group2', f'{i:0>2}.tfrecord', 2)
        create_tfrecord('tma', 4, data_dir, output_dir + r'\Group3', f'{i:0>2}.tfrecord', 3)
        create_tfrecord('tma', 4, data_dir, output_dir + r'\Group4', f'{i:0>2}.tfrecord', 4)
