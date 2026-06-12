# Copyright (c) Open-CD. All rights reserved.
import os
from PIL import Image
import shutil
import numpy as np

def calculate_change_ratio(label_image_path):

    label_image = Image.open(label_image_path).convert('L')

    label_array = np.array(label_image)

    total_pixels = label_array.size
    change_pixels = np.sum(label_array == 255)
    change_ratio = change_pixels / total_pixels

    return change_ratio

def create_output_directories(base_dir):

    os.makedirs(os.path.join(base_dir, 'A'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'B'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'label'), exist_ok=True)

def copy_files_to_directory(image_name, source_dir, target_dir):

    shutil.copy(os.path.join(source_dir, 'A', image_name), os.path.join(target_dir, 'A', image_name))

    shutil.copy(os.path.join(source_dir, 'B', image_name), os.path.join(target_dir, 'B', image_name))

    shutil.copy(os.path.join(source_dir, 'label', image_name), os.path.join(target_dir, 'label', image_name))

def partition_images(source_dir, s_dir, m_dir, l_dir, threshold1=0.05, threshold2=0.2):

    label_images = sorted(os.listdir(os.path.join(source_dir, 'label')))

    for image_name in label_images:
        label_image_path = os.path.join(source_dir, 'label', image_name)

        change_ratio = calculate_change_ratio(label_image_path)

        if change_ratio <= threshold1:
            target_dir = s_dir
        elif threshold1 < change_ratio <= threshold2:
            target_dir = m_dir
        else:
            target_dir = l_dir

        copy_files_to_directory(image_name, source_dir, target_dir)

def process_dataset(base_dir, dataset_type):

    source_dir = os.path.join(base_dir, dataset_type)
    s_dir = os.path.join(base_dir, f'{dataset_type}_s')
    m_dir = os.path.join(base_dir, f'{dataset_type}_m')
    l_dir = os.path.join(base_dir, f'{dataset_type}_l')

    create_output_directories(s_dir)
    create_output_directories(m_dir)
    create_output_directories(l_dir)

    partition_images(source_dir, s_dir, m_dir, l_dir)

def main():

    base_dir = 'data/JL1-CD'

    for dataset_type in ['train', 'val', 'test']:
        process_dataset(base_dir, dataset_type)

if __name__ == '__main__':
    main()
