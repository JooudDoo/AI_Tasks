import os

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
    fig = plt.figure(figsize=(20, 10))
    for idx, (name, image) in enumerate(images.items()):
        ax = fig.add_subplot(idx//row_len+1, n_images, idx+1)
        ax.set_title(name.title(), fontsize=16)
        ax.imshow(image)
    return fig