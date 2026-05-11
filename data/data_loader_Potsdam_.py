import numpy as np
import os
import glob
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
pjoin = os.path.join
import random
from skimage import io

class ifly_train(data.Dataset):
    def __init__(self, img_path, edge_path, transform):
        self.transform = transform
        self.img_path = img_path
        self.edge_path = edge_path
        self.WINDOW_SIZE = (256, 256)
        self.cache = True
        self.augmentation = True
        # List of files
        # self.data_files = self.img_path
        # self.label_files = self.edge_path
        self.data_files = [id for id in self.img_path]
        self.label_files = [id for id in self.edge_path]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}

        self.transform1 =  transforms.Compose([
        transforms.ToTensor() ])


    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, index):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            ## Vaihingen IRRG
            data = io.imread(self.data_files[random_idx])
            data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(io.imread(self.label_files[random_idx]), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, self.WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))

    def __len__(self):
        return 10000

class ifly_test(data.Dataset):
    def __init__(self, img_path, lab_path, transform):
        self.transform = transform
        self.img_path = img_path
        self.lab_path = lab_path
        self.transform1 =  transforms.Compose([
        transforms.ToTensor() ])

    def __getitem__(self, index):
        img = np.array(Image.open(self.img_path[index]))
        img = Image.fromarray(img)
        img = self.transform(img)
        lab = np.array(Image.open(self.lab_path[index])).astype(np.int32)
        # lab = Image.fromarray(lab)
        lab = torch.tensor(lab, dtype=torch.float32)
        # lab = self.transform1(lab)
        return img.squeeze(0), lab

    def __len__(self):
        return len(self.img_path)

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def get_data_loader(root_path):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    # train datasets
    img_path = root_path + '/train/images'
    lab_path = root_path + '/train/labels'
    train_img = glob.glob(img_path + '/*.tif')
    train_lab = glob.glob(lab_path + '/*.png')
    train_img.sort(key=lambda x: x.split('/')[-1].split('.tif')[0])
    train_lab.sort(key=lambda x: x.split('/')[-1].split('.png')[0])
    train_image_path = train_img
    train_lab_path = train_lab

    # val datasets
    img_path = root_path + '/val/images'
    lab_path = root_path + '/val/labels'
    test_img = glob.glob(img_path + '/*.png')
    test_lab = glob.glob(lab_path + '/*.png')
    test_img.sort(key=lambda x: x.split('/')[-1].split('.png')[0])
    test_lab.sort(key=lambda x: x.split('/')[-1].split('.png')[0])

    test_image_path = test_img
    test_lab_path = test_lab
    train_set = ifly_train(train_image_path, train_lab_path, transform_train)
    test_set = ifly_test(test_image_path, test_lab_path,transform_test)

    return train_set, test_set