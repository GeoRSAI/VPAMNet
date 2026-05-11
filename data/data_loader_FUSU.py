import numpy as np
import os
import glob
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
pjoin = os.path.join


class ifly_train(data.Dataset):
    def __init__(self, img_path, edge_path, transform):
        self.transform = transform
        self.img_path = img_path
        self.edge_path = edge_path
        self.transform1 =  transforms.Compose([
        transforms.ToTensor() ])

    def __getitem__(self, index):
        img = np.array(Image.open(self.img_path[index]))
        img = Image.fromarray(img)
        img = self.transform(img)
        lab = np.array(Image.open(self.edge_path[index]))
        # lab = Image.fromarray(lab)
        lab = torch.tensor(lab, dtype=torch.float32)
        # lab = self.transform1(lab)
        # augmented = self.augmentation(image=img, masks=[edge])
        # img = augmented['image']
        # edge = augmented['masks'][0]/255.
        # lab = torch.stack((lab, edge), dim=0)
        return img.squeeze(0),lab

    def __len__(self):
        return len(self.img_path)

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
        lab = np.array(Image.open(self.lab_path[index]))
        # lab = Image.fromarray(lab)
        lab = torch.tensor(lab, dtype=torch.float32)
        # lab = self.transform1(lab)
        return img.squeeze(0), lab

    def __len__(self):
        return len(self.img_path)

def get_data_loader(root_path):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    ###
    img_path = root_path + '/img1'
    lab_path = root_path + '/label2'
    train_img = glob.glob(img_path + '/*.png')
    train_lab = glob.glob(lab_path + '/*.png')
    train_img.sort(key=lambda x: x.split('/')[-1].split('.png')[0])
    train_lab.sort(key=lambda x: x.split('/')[-1].split('.png')[0])
    np.random.seed(0)
    np.random.shuffle(train_img)
    np.random.seed(0)
    np.random.shuffle(train_lab)
    train_image_path = train_img[:10000]
    train_lab_path = train_lab[:10000]
    test_image_path = train_img[10000:14000]
    test_lab_path = train_lab[10000:14000]

    # # train datasets
    # img_path = root_path + '/train/images'
    # lab_path = root_path + '/train/labels'
    # train_img = glob.glob(img_path + '/*.png')
    # train_lab = glob.glob(lab_path + '/*.png')
    # train_img.sort(key=lambda x: x.split('/')[-1].split('.png')[0])
    # train_lab.sort(key=lambda x: x.split('/')[-1].split('.png')[0])
    # train_image_path = train_img
    # train_lab_path = train_lab
    #
    # # val datasets
    # img_path = root_path + '/val/images'
    # lab_path = root_path + '/val/labels'
    # test_img = glob.glob(img_path + '/*.png')
    # test_lab = glob.glob(lab_path + '/*.png')
    # test_img.sort(key=lambda x: x.split('/')[-1].split('.png')[0])
    # test_lab.sort(key=lambda x: x.split('/')[-1].split('.png')[0])
    #
    # test_image_path = test_img
    # test_lab_path = test_lab
    train_set = ifly_train(train_image_path, train_lab_path, transform_train)
    test_set = ifly_test(test_image_path, test_lab_path,transform_test)

    return train_set, test_set