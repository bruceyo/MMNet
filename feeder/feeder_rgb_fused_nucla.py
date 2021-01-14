# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools

# rgb --B
from PIL import Image
from feeder import segment_rgbbody_nucla as rgb_roi

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 centralization=False,
                 window_size=-1,
                 part='train',
                 debug=False,
                 mmap=True,
                 random_interval=False,
                 random_roi_move=False,
                 temporal_rgb_frames=5,
                 evaluation=False,
                 random_flip=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.centralization = centralization
        self.window_size = window_size
        self.part = part
        if self.part=='train':
            self.evaluation = False
        else:
            self.evaluation = True

        self.random_interval = random_interval
        self.random_roi_move = random_roi_move
        self.random_flip = random_flip
        self.temporal_rgb_frames = temporal_rgb_frames
        #self.rgb_path = '/mnt/nas/ntu-rgbd/NTU/RGB_videos/rgb_frames/'
        self.rgb_path = '../../data/NWestern_UCLA/ucla_rgb_frames_crop/fivefs_240/'

        self.load_data(mmap)
        if self.part=='train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(225,45*self.temporal_rgb_frames)), #(h,w)
                #transforms.ColorJitter(hue=.05, saturation=.05),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(20, resample=Image.BILINEAR),
                #transforms.RandomErasing(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                #transforms.Normalize(mean=[0.634, 0.645, 0.675], std=[0.264, 0.260, 0.274])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(225,45*self.temporal_rgb_frames)),
                #transforms.ColorJitter(hue=.05, saturation=.05),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(20, resample=Image.BILINEAR),
                #transforms.RandomErasing(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                #transforms.Normalize(mean=[0.634, 0.645, 0.675], std=[0.264, 0.260, 0.274])
            ])

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # add RGB features based on self.sample_name  -- B
        # print('self.sample_name',self.sample_name)
        #rgb = self.sample_name[index][0:12] + '.png'
        #rgb = Image.open(self.rgb_path + rgb)
        rgb = rgb_roi.construct_st_roi(self.sample_name[index][0:12], self.evaluation, self.random_interval,self.random_roi_move,self.random_flip, self.temporal_rgb_frames)
        width, height = rgb.size

        rgb = np.array(rgb.getdata())
        #rgb_origin = rgb
        rgb = torch.from_numpy(rgb).float()
        T, C = rgb.size()
        #if T != 162*162:
        #    print('Size Error',self.sample_name[index][0:20])

        rgb = rgb.permute(1, 0).contiguous()
        #rgb = rgb.view(C, width, height)
        rgb = rgb.view(C, height, width)
        #print('rgb.size(): ',rgb.size())
        rgb = self.transform(rgb)
        #print('rgb.size() after transform: ',rgb.size())
        if self.centralization:
            data_numpy = tools.centralization(data_numpy,rotate=True, part=self.part)
        if self.part=='train':
            # processing
            if self.random_choose:
                data_numpy = tools.random_choose(data_numpy, self.window_size)
            elif self.window_size > 0:
                data_numpy = tools.auto_pading(data_numpy, self.window_size)
            if self.random_move:
                data_numpy = tools.random_move(data_numpy)

        return data_numpy, rgb, label#, rgb_origin
