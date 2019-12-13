import os
import os.path
import glob

import torch
import torch.utils.data as data
import numpy as np
import random
import cv2


def get_loader(is_train, root, mv_dir):
    print('\nCreating loader for %s...' % root)

    dset = ImageFolder(
        is_train=is_train,
        root=root,
        mv_dir=mv_dir
    )

    loader = data.DataLoader(
        dataset=dset,
        batch_size=1,
        shuffle=is_train,
        num_workers=2
    )

    print('Loader for {} images ({} batches) created.'.format(
        len(dset), len(loader))
    )

    return loader


def default_loader(path):
    cv2_img = cv2.imread(path)
    if cv2_img.shape is None:
        print(path)
        print(cv2_img)
    else:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    width, height, _ = cv2_img.shape
    if width % 16 != 0 or height % 16 != 0:
        cv2_img = cv2_img[:(width//16)*16, :(height//16)*16]

    return cv2_img

def crop_cv2(img, patch):
    height, width, c = img.shape
    start_x = 0#random.randint(0, height - patch)
    start_y = 0#random.randint(0, width - patch)

    return img[start_x : start_x + patch, start_y : start_y + patch]


def flip_cv2(img, patch):
    if random.random() < 0.5:
        img = img[:, ::-1, :].copy()

        assert img.shape[2] == 13, img.shape
        # height first, and then width. but BMV is (width, height)... sorry..
        img[:, :, 9] = img[:, :, 9] * (-1.0)
        img[:, :, 11] = img[:, :, 11] * (-1.0)
    return img


def np_to_torch(img):
    img = np.swapaxes(img, 0, 1) #w, h, 9
    img = np.swapaxes(img, 0, 2) #9, h, w
    return torch.from_numpy(img).float()


class ImageFolder(data.Dataset):
    def __init__(self, is_train, root, mv_dir):

        self.is_train = is_train
        self.root = root

        self.loader = default_loader
        self._load_image_list()

    def _load_image_list(self):
        self.imgs = []
        for filename in glob.iglob(self.root + '/*png'):
            self.imgs.append(filename)

        print('%d images loaded.' % len(self.imgs))

    def __getitem__(self, index):
        if index > 3:
            index -= 2
        filenames = [self.imgs[index], self.imgs[index+1], self.imgs[index+2]]
        img = [self.loader(filename) for filename in filenames]
        img = np.array(img)/255.0
        data = np_to_torch(img)

        return data, filenames

    def __len__(self):
        return len(self.imgs)
