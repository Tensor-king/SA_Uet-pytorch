import numpy as np
from PIL import Image

# np.set_printoptions(threshold=10000000000)
path1 = r"D:\Deep-Learning\deep-learning-for-image-processing-master\pytorch_segmentation\SA_Uet\CHASEDB1\aug\1st_label\randomColor0Image_08L_1stHO.png"
img1 = Image.open(path1)
print(img1)
x1 = np.array(img1)
print(x1[500][501])
print(x1)
print(x1.shape)
a = np.max(x1)
print(a)
