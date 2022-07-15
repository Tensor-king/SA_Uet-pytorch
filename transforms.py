import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)  # float类型
        target = torch.as_tensor(np.array(target), dtype=torch.int64)  # long类型
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        target = (target / 255).long()
        return image, target


class RandomRotation2022(object):
    def __init__(self):
        self.angle = random.randint(1, 360)

    def __call__(self, image, target, label):
        image = T.RandomRotation((self.angle, self.angle), resample=Image.BICUBIC)(image)
        target = T.RandomRotation((self.angle, self.angle), resample=Image.NEAREST)(target)
        label = T.RandomRotation((self.angle, self.angle), resample=Image.NEAREST)(label)
        return image, target, label


class ColorJitter(object):

    def __call__(self, image):
        image = T.ColorJitter(brightness=(1, 2), contrast=(
            1, 2), saturation=(1, 2), hue=(-0.5, 0.5))(image)
        return image


def randomColor(image):
    random_factor = np.random.randint(0, 31) / 10.
    color_image = ImageEnhance.Color(image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(0, 31) / 10.
    sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
    return sharp_image


def gaussianNoisy(im, mean, sigma):
    for _i in range(len(im)):
        im[_i] += random.gauss(mean, sigma)
    return im


class RandomGaussian(object):
    def __init__(self, mean=0.2, sigma=0.3):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, image):
        # H,W,C
        img = np.asarray(image)
        width, height = img.shape[0], img.shape[1]

        img_r = gaussianNoisy(img[:, :, 0].flatten(), self.mean, self.sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), self.mean, self.sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), self.mean, self.sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))


if __name__ == "__main__":
    train_path = r"D:\Deep-Learning\Seg-datasets\DRIVE\training\images"
    ground_path = r"D:\Deep-Learning\Seg-datasets\DRIVE\training\1st_manual"
    mask_path = r"D:\Deep-Learning\Seg-datasets\DRIVE\training\mask"
    # 接下来进行数据增强操作
    aug_option = {"RandomRandomRotation": RandomRotation2022(), "RandomGaussian": RandomGaussian(),
                  "ColorJitter": randomColor,
                  }

    train_path_list = [os.path.join(train_path, i) for i in os.listdir(train_path)]
    ground_path_list = [os.path.join(ground_path, i) for i in os.listdir(ground_path)]
    mask_path_list = [os.path.join(mask_path, i) for i in os.listdir(mask_path)]

    if not os.path.exists(os.path.join("Aug_Drive", "training", "images")):
        os.makedirs(os.path.join("Aug_Drive", "training", "images"))
    if not os.path.exists(os.path.join("Aug_Drive", "training", "1st_manual")):
        os.makedirs(os.path.join("Aug_Drive", "training", "1st_manual"))
    if not os.path.exists(os.path.join("Aug_Drive", "training", "mask")):
        os.makedirs(os.path.join("Aug_Drive", "training", "mask"))

    count = 0
    for k in range(3):
        for i in range(len(train_path_list)):
            for j in range(4):
                label = Image.open(ground_path_list[i])
                mask = Image.open(mask_path_list[i])
                img = cv2.imread(train_path_list[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                img = Image.fromarray(np.uint8(img))
                if not j == 0:
                    if k == 0:
                        transformer = RandomRotation2022()
                        img, label, mask = transformer(img, label, mask)
                    elif k == 1:
                        transformer = RandomGaussian()
                        img = transformer(img)
                    elif k == 2:
                        transformer = randomColor
                        img = transformer(img)
                count += 1
                img.save(os.path.join("Aug_Drive", "training", "images", f"{count}" + "_training.tif"))
                label.save(os.path.join("Aug_Drive", "training", "1st_manual", f"{count}" + "_manual.gif"))
                mask.save(os.path.join("Aug_Drive", "training", "mask", f"{count}" + "_training_mask.gif"))
