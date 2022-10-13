import os

from PIL import Image
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        data_root = os.path.join(root, "aug" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "labels", i) for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        # 值为0和255
        mask = Image.open(self.manual[idx]).convert('L')
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
            # gray_img = img[0, :, :] * 0.299 + img[1, :, :] * 0.587 + img[2, :, :] * 0.114
            # gray_img = gray_img.reshape(1, img.shape[1], img.shape[2])
        return img, mask

    def __len__(self):
        return len(self.img_list)


class Chasedb1Datasets:
    def __init__(self, root: str, train: bool, transforms=None):
        super().__init__()
        data_root = os.path.join(root, "aug" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "labels", i)
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        # 注意：chansedb1的label的原位深为1，后被我转换为8，即0表示背景，255表示血管
        label = Image.open(self.manual[idx]).convert('L')
        if self.transforms is not None:
            img, mask = self.transforms(img, label)

        return img, label

    def __len__(self):
        return len(self.img_list)


class StareDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(StareDataset, self).__init__()
        data_root = os.path.join(root, "aug" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "labels", i) for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        # 值为0和255
        mask = Image.open(self.manual[idx]).convert('L')
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
            # gray_img = img[0, :, :] * 0.299 + img[1, :, :] * 0.587 + img[2, :, :] * 0.114
            # gray_img = gray_img.reshape(1, img.shape[1], img.shape[2])
        return img, mask

    def __len__(self):
        return len(self.img_list)
