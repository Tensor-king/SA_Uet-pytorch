import numpy as np
import os
from PIL import Image


def compute():
    img_channels = 3
    img_dir = "DRIVE/aug/images"
    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".tif")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        # img转化为0到1
        img = np.array(Image.open(img_path)) / 255.
        cumulative_mean += img.mean(axis=(0, 1))
        cumulative_std += img.std(axis=(0, 1))

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    return mean, std
