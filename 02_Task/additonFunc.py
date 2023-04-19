import os
from os.path import join as pjoin
import contextlib
import sys

import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def uniqufy_path(path):
    filename, extension = os.path.splitext(path)
    file_index = 1

    while os.path.exists(path):
        path = f"{filename}_{file_index}{extension}"
        file_index += 1
    
    return path

def create_image_plot(row_len : int = None, **images):
    n_images = len(images)
    if row_len is None:
        row_len = n_images
    fig = plt.figure(figsize=(16, 6))
    for idx, (name, image) in enumerate(images.items()):
        ax = fig.add_subplot(idx//row_len+1, n_images, idx+1)
        ax.set_title(name.title(), fontsize=16)
        with open("/dev/null", 'w') as dummy_f:
            with contextlib.redirect_stderr(dummy_f):
                ax.imshow(image)
    return fig

def save_imgs(path = None, name = "imgs", **images):
    if(path is None):
        raise AttributeError(f"You shoud write path")
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = pjoin(path, f"{name}")
    fig = create_image_plot(**images)
    fig.savefig(image_path)
    fig.clear()
    plt.close(fig)
