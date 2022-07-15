import numpy as np
from PIL import Image

path = r"/tmp/pycharm_project_452/DRIVE/training/images/23_training.tif"
path = Image.open(path)
x = np.array(path)
print(x.shape)
print(x.mean((0, 1)).shape)
