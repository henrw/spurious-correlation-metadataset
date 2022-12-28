import os

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
import inspect
import json
from sklearn.utils import shuffle, resample

class MNIST_CIFAR10():
    """
    MNIST CIFAR10 dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
      root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
      env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
      ratio (list): train:val:test ratio
      transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """

    def __init__(self, args, root='./data/', ratio = [0.6, 0.2, 0.2], transform=None, target_transform=None):
        # super(ColoredMNIST, self).__init__(root, transform=transform,
        #                                    target_transform=target_transform)
        for i in args.flip_ratio_list:
            assert i >= 0 and i <= 1
        assert args.num_class == len(args.flip_ratio_list)
        self.args = args
        self.root = root
        with open("./mnist_cifar10/metadata.json", 'r') as f:
            self.meta = json.load(f)

        # set file_name for dataset csv:
        self.filestr = f"num_class_{str(args.num_class)}_flip_ratio_{'_'.join([str(item) for item in args.flip_ratio_list])}_class_ratio_{str(args.class_ratio)}"

        self.prepare_cifar10_mnist()

    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index]

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target

    # def __len__(self):
    #     return len(self.data)

    # def read_csv_file(self, data_csv):
    #     # todo
    #     data = data_csv
    #     return data

    def prepare_cifar10_mnist(self):
        # check whether the csv file exists
        if os.path.exists(os.path.join(self.root, self.filestr + '_metadata.csv')):
            print(
                f'The required MNIST CIFAR10 dataset already exists:\n{self.filestr}')
            # # read csv file
            # data_csv = pd.read_csv(
            #     os.path.join(self.root, self.filename)
            # )
            # self.data = self.read_csv_file(data_csv)
            return

        print('Preparing MNIST CIFAR10')

        # all the data needed
        info_dict = {
            'idx': [],
            'label': [],
            'binary_label': [],
            'spurious_label': [],
            'flipped': [],
        }

        with open(self.meta["files"]["train"]["car"], 'rb') as train_car_f:
            train_car = np.load(train_car_f)
        with open(self.meta["files"]["train"]["truck"], 'rb') as train_truck_f:
            train_truck = np.load(train_truck_f)
        with open(self.meta["files"]["train"]["0"], 'rb') as train_0_f:
            train_0 = np.load(train_0_f)
        with open(self.meta["files"]["train"]["1"], 'rb') as train_1_f:
            train_1 = np.load(train_1_f)

        train_num_half = min(self.meta["total_num"]["train"].values())
        train_num = 2 * train_num_half
        train_car = resample(torch.tensor(train_car), n_samples=train_num_half)
        train_truck = resample(torch.tensor(train_truck), n_samples=train_num_half)
        train = torch.concat([train_car,train_truck])
        train_labels = torch.hstack([torch.zeros((train_num_half),dtype=torch.uint8), torch.ones((train_num_half),dtype=torch.uint8)])
        
        train_0 = resample(torch.tensor(train_0), n_samples=train_num_half)
        train_1 = resample(torch.tensor(train_1), n_samples=train_num_half)
        train_spurious = torch.concat([train_0,train_1])
        train_spurious_labels = torch.hstack([torch.zeros((train_num_half),dtype=torch.uint8), torch.ones((train_num_half),dtype=torch.uint8)])

        train, train_spurious, train_labels, train_spurious_labels = shuffle(
            train, train_spurious, train_labels, train_spurious_labels
        )

        with open(self.meta["files"]["test"]["car"], 'rb') as test_car_f:
            test_car = np.load(test_car_f)
        with open(self.meta["files"]["test"]["truck"], 'rb') as test_truck_f:
            test_truck = np.load(test_truck_f)
        with open(self.meta["files"]["test"]["0"], 'rb') as test_0_f:
            test_0 = np.load(test_0_f)
        with open(self.meta["files"]["test"]["1"], 'rb') as test_1_f:
            test_1 = np.load(test_1_f)
        
        test_num_half = min(self.meta["total_num"]["test"].values())
        test_num = 2 * test_num_half

        test_car = resample(torch.tensor(test_car), n_samples=test_num_half)
        test_truck = resample(torch.tensor(test_truck), n_samples=test_num_half)
        test = torch.concat([test_car,test_truck])
        test_labels = torch.hstack([torch.zeros((test_num_half),dtype=torch.uint8), torch.ones((test_num_half),dtype=torch.uint8)])
        
        test_0 = resample(torch.tensor(test_0), n_samples=test_num_half)
        test_1 = resample(torch.tensor(test_1), n_samples=test_num_half)
        
        flip_ratio = self.args.flip_ratio_list[0]
        sep_idx = int(flip_ratio * test_num_half)
        test_spurious = torch.concat([test_1[:sep_idx],test_0[sep_idx:], test_0[:sep_idx],test_1[sep_idx:]])
        test_spurious_labels = torch.tensor([1]*sep_idx+[0]*test_num_half+[1]*(test_num_half-sep_idx),dtype=torch.uint8)
        flipped_arr = ([True]*sep_idx + [False]*(test_num_half-sep_idx))*2
        test, test_spurious, test_labels, test_spurious_labels, flipped_arr = shuffle(
            test, test_spurious, test_labels, test_spurious_labels, flipped_arr
        )

        self.train_img = torch.concat([train_spurious, train], dim=1)
        self.train_labels = train_labels
        self.train_spurious_labels = train_spurious_labels
        self.test_img = torch.concat([test_spurious, test], dim=1)
        self.test_labels = test_labels
        self.test_spurious_labels = test_spurious_labels
        self.flipped_arr = flipped_arr
        
        img_all = torch.concat([self.train_img, self.test_img], dim=0)

        # iteration
        for idx, (label, spurious_label) in enumerate(zip(train_labels, train_spurious_labels)):
            flipped = False

            info_dict['idx'].append(idx)
            info_dict['label'].append(self.meta["labels"][label])
            info_dict['binary_label'].append(label.item())
            info_dict['spurious_label'].append(spurious_label.item())
            info_dict['flipped'].append(int(flipped))
        
        for idx, (label, spurious_label, flipped) in enumerate(zip(test_labels, test_spurious_labels, flipped_arr)):

            info_dict['idx'].append(idx)
            info_dict['label'].append(self.meta["labels"][label])
            info_dict['binary_label'].append(label.item())
            info_dict['spurious_label'].append(spurious_label.item())
            info_dict['flipped'].append(int(flipped))

        info_df = pd.DataFrame()
        for key in info_dict.keys():
            info_df[key] = info_dict[key]

    
        train_cnt, val_cnt = train_num * 3 // 4, train_num // 4
        split_list = [0] * int(train_cnt) + [1] * int(val_cnt)
        np.random.shuffle(split_list)
        split_list += [2] * test_num
        info_df['split'] = split_list

        os.makedirs(self.root, exist_ok=True)
        torch.save(
            img_all,
            os.path.join(self.root, self.filestr + '_imgdata.pt')
        )
        
        info_df.to_csv(
            os.path.join(self.root, self.filestr + '_metadata.csv')
        )

        print("DONE")