import os
from os.path import join as pjoin
import shutil

import cv2
import numpy as np

from tqdm import tqdm

def extract_filename(filepath : str):
    return filepath.split('/')[-1]

def calculate_padding(orig, pad):
    xPad = pad[0] - orig[0]
    yPad = pad[1] - orig[1]
    return (int(xPad/2), int(xPad/2), int(yPad/2), int(yPad/2))

DATASET_DIR = 'dataset/tiff'

TARGET_DIRS = [('test', 'test_labels'), ('val', 'val_labels')]

CROP_SIZE = (256,256)
PAD_SIZE  = (1536, 1536)

MODE = "CROP_COPY_DEL" # "CROP" | "COPY" | "DEL"

COPY_FOLDER_SUFFIX = "_non_cutted"

print(f"Начата нарезка датасета: {DATASET_DIR}")
print(f"Будут осмотрены папки внутри основной директории: {TARGET_DIRS}")
print(f"Текущий режим работы: {MODE}")


for id, target in enumerate(TARGET_DIRS):
    origin_dir = pjoin(DATASET_DIR, target[0])
    masks_dir  = pjoin(DATASET_DIR, target[1])

    origins_names = [pjoin(origin_dir, filename) for filename in sorted(os.listdir(origin_dir))]
    masks_names   = [pjoin(masks_dir, filename) for filename in sorted(os.listdir(masks_dir))]

    image_names = list(zip(origins_names, masks_names))
    for origin_path, mask_path in (pbar := tqdm(image_names)):
        origin = cv2.imread(origin_path)
        mask = cv2.imread(mask_path)
        h, w, c = origin.shape
        origin = cv2.copyMakeBorder(origin, *calculate_padding((h,w), PAD_SIZE), cv2.BORDER_CONSTANT, None, value = (255, 255, 255))
        mask = cv2.copyMakeBorder(mask,   *calculate_padding((h,w), PAD_SIZE), cv2.BORDER_CONSTANT, None, value = 0)
        h, w, c = origin.shape

        if("COPY" in MODE):
            origin_copy_folder = pjoin(DATASET_DIR, f"{target[0]}{COPY_FOLDER_SUFFIX}")
            mask_copy_folder = pjoin(DATASET_DIR, f"{target[1]}{COPY_FOLDER_SUFFIX}")

            if not os.path.exists(origin_copy_folder):
                os.makedirs(origin_copy_folder)

            if not os.path.exists(mask_copy_folder):
                os.makedirs(mask_copy_folder)
            
            shutil.copyfile(origin_path, pjoin(origin_copy_folder, extract_filename(origin_path)))
            shutil.copyfile(mask_path,   pjoin(mask_copy_folder, extract_filename(mask_path)))

        if("CROP" in MODE):
            origin_parts = []
            mask_parts = []
            for offset_x in range(0, w, CROP_SIZE[0]):
                for offset_y in range(0, h, CROP_SIZE[1]):
                    origin_parts.append(origin[offset_y:offset_y+CROP_SIZE[1], offset_x:offset_x+CROP_SIZE[0]])
                    mask_parts.append(mask[offset_y:offset_y+CROP_SIZE[1], offset_x:offset_x+CROP_SIZE[0]])
            origin_filename, origin_file_extension = os.path.splitext(origin_path)
            mask_filename, mask_file_extension = os.path.splitext(mask_path)
            for id, part in enumerate(zip(origin_parts, mask_parts)):
                cv2.imwrite(f"{origin_filename}_{id}{origin_file_extension}", part[0])
                cv2.imwrite(f"{mask_filename}_{id}{mask_file_extension}", part[1])
            
        if ("DEL" in MODE):
                os.remove(origin_path)
                os.remove(mask_path)
        pbar.set_description(f"Обработка изображений в {target[0]} и {target[1]}")