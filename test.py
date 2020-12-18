# -*- coding: utf-8 -*-
"""

"""
import os
import random
import numpy as np
import torch

from torch.utils.data.dataloader import DataLoader
from monet_dataloader import MonetDataset
from options.test_options import TestOptions
from models.monet_model import StyleTransfer


if __name__ == '__main__':
    opt = TestOptions().parse()
    if not os.path.isdir(opt.results_dir):
        os.mkdir(opt.results_dir)
    
    test_dataset = MonetDataset(opt.data_dir,train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = StyleTransfer(opt)
    
    opt.start_epoch = model.load(opt.load_epoch,opt.checkpoints_dir)
    print("model loaded from:",opt.start_epoch)
    print("start testing")
    total_steps = 0
    for ii, batch in enumerate(test_dataloader):
        
        total_steps += 1
        
        _ = model.test(batch)
        _, _, _ = model.save_img(opt.results_dir, total_steps)
        
        if total_steps % opt.print_freq == 0:
            print('processing...',total_steps)
            
    print('Finished testing!')
            
