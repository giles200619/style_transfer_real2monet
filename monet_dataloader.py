# -*- coding: utf-8 -*-
"""

"""

import random
import sys
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MonetDataset(Dataset):
    def __init__(self, path, train=True):
        self.train = train
        if train:
            assert(os.path.isdir(os.path.join(path,'trainA')))
            assert(os.path.isdir(os.path.join(path,'trainB')))
            self.path_A = os.path.join(path,'trainA')
            self.path_B = os.path.join(path,'trainB')
        else:
            assert(os.path.isdir(os.path.join(path,'testA')))
            assert(os.path.isdir(os.path.join(path,'testB')))
            self.path_A = os.path.join(path,'testA')
            self.path_B = os.path.join(path,'testB')
            
        self.data_list_A = os.listdir(self.path_A)
        self.data_list_B = os.listdir(self.path_B)
            
        self.length = len(self.data_list_A)
        self.length_B = len(self.data_list_B)
        
        self.trans = transforms.Compose([
    	transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    	])
        
    def __getitem__(self, idx):
        
        imgA = Image.open(os.path.join(self.path_A, self.data_list_A[idx]))
        if self.train:
            idx_B = random.randint(0,self.length_B-1)
            imgB = Image.open(os.path.join(self.path_B, self.data_list_B[idx_B]))
        else:
            idx_B = idx if idx<self.length_B else random.randint(0,self.length_B-1)
            imgB = Image.open(os.path.join(self.path_B, self.data_list_B[idx_B]))
        
        data = {}
        data['content'] = self.trans(imgA)
        data['style'] = self.trans(imgB)
        
        return data

    def __len__(self):
        return self.length


