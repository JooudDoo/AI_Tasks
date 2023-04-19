import os
from os.path import join as pjoin
import shutil

import cv2
import numpy as np

from tqdm import tqdm

def extract_filename(filepath : str):
    return filepath.split('/')[-1]

DATASET_DIR = 'dataset/tiff'

TARGET_DIRS = [('train', 'train_labels'), ('test', 'test_labels'), ('val', 'val_labels')]

THRESHOLD = 0.4

REMOVE_MODE = "FIX_MASKS_COPY" # "DEL" | "COPY_DEL" | "FIX_MASKS" | "COPY"

print(f"Начата очистка: {DATASET_DIR}")
print(f"Будут осмотрены папки внутри основной директории: {TARGET_DIRS}")
print(f"Текущий режим работы: {REMOVE_MODE}")

for id, target in enumerate(TARGET_DIRS):
    origin_dir = pjoin(DATASET_DIR, target[0])
    masks_dir  = pjoin(DATASET_DIR, target[1])

    origins_names = [pjoin(origin_dir, filename) for filename in sorted(os.listdir(origin_dir))]
    masks_names   = [pjoin(masks_dir, filename) for filename in sorted(os.listdir(masks_dir))]

    image_names = list(zip(origins_names, masks_names))
    broken_pair_count = 0
    for origin_path, mask_path in (pbar := tqdm(image_names)):
        origin = cv2.imread(origin_path)
        pixels_count = np.prod(origin.shape[:2])
        pixels_white_count = np.sum(origin == [255, 255, 255])
        white_pixels_ratio = pixels_white_count / pixels_count
        if(white_pixels_ratio > THRESHOLD):
            broken_pair_count += 1

            if("COPY" in REMOVE_MODE):
                origin_copy_folder = pjoin(DATASET_DIR, f"{target[0]}_to_remove")
                mask_copy_folder = pjoin(DATASET_DIR, f"{target[1]}_to_remove")

                if not os.path.exists(origin_copy_folder):
                    os.makedirs(origin_copy_folder)

                if not os.path.exists(mask_copy_folder):
                    os.makedirs(mask_copy_folder)
                
                shutil.copyfile(origin_path, pjoin(origin_copy_folder, extract_filename(origin_path)))
                shutil.copyfile(mask_path,   pjoin(mask_copy_folder, extract_filename(mask_path)))

            if("FIX_MASKS" in REMOVE_MODE):
                mask = cv2.imread(mask_path)
                mask[origin == [255, 255, 255]] = 0
                cv2.imwrite(mask_path, mask)
            
            if ("DEL" in REMOVE_MODE):
                os.remove(origin_path)
                os.remove(mask_path)
        pbar.set_description(f"Обработка изображений в {target[0]} и {target[1]} | Изменено пар: {broken_pair_count}")
