# -*- coding: utf-8 -*-
"""

"""
import os
import torch
import numpy as np
from PIL import Image
from .generator import Generator
from .discriminator import BaseDiscriminator
from .losses.perceptual_loss import ContentLoss, StyleLoss
from .layers.helpers import initialize_model, data_parallel


class StyleTransfer():
    def name(self,):
        return 'StyleTransfer'
    
    def __init__(self,opt):
        self.opt = opt
        self.device = opt.device
        
        self.generator = Generator(opt)
        self.generator = initialize_model(self.generator, opt)
        self.generator = data_parallel(self.generator, self.opt)
        
        if opt.phase == 'train':  
            self.discriminator = BaseDiscriminator(opt)
            self.discriminator = initialize_model(self.discriminator, opt)
            self.discriminator = data_parallel(self.discriminator, self.opt)
            
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001)
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
          
            num_params = 0
            for param in self.generator.parameters():
                num_params += param.numel()
            print('[Network %s] Total number of parameters : %.3f M' % ('Generator', num_params / 1e6))
            num_params = 0
            for param in self.discriminator.parameters():
                num_params += param.numel()
            print('[Network %s] Total number of parameters : %.3f M' % ('Discriminator', num_params / 1e6))
            
            if opt.lambda_content > 0.0:
                self.criterion_content = data_parallel(ContentLoss(), opt)
            
            if opt.lambda_style > 0.0:
                self.criterion_style = data_parallel(StyleLoss(), opt)
                
            self.lambda_GAN = opt.lambda_GAN
            self.lambda_content = opt.lambda_content
            self.lambda_style = opt.lambda_style

    def set_input(self, input):
        
        self.content = input['content'].to(self.device)
        self.style = input['style'].to(self.device) 
     
    def save(self,epoch,latest=False):
        path = self.opt.checkpoints_dir
        if latest:
            save_filename = 'monet_%s.pth' % ('latest')
        else:
            save_filename = 'monet_%s.pth' % (epoch)
        save_path = os.path.join(path, save_filename)
        data = {}
        data['epoch'] = epoch
        if len(self.opt.gpu_ids) > 1 and torch.cuda.is_available:
            data['G'] = self.generator.module.state_dict()
            data['D'] = self.discriminator.module.state_dict()
        else:
            data['G'] = self.generator.state_dict()
            data['D'] = self.discriminator.state_dict()
        data['optimizer_G'] = self.optimizer_G.state_dict()
        data['optimizer_D'] = self.optimizer_D.state_dict()
        torch.save(data, save_path)
        
    def load(self,epoch,path):
        filename = 'monet_%s.pth' % (epoch)
        path = os.path.join(path, filename)
        checkpoint = torch.load(path)
        if self.opt.phase == 'train':  
            if isinstance(self.generator, torch.nn.DataParallel):
                self.generator.module.load_state_dict(checkpoint['G'])
                self.discriminator.module.load_state_dict(checkpoint['D'])
            else:
                self.generator.load_state_dict(checkpoint['G'])
                self.discriminator.load_state_dict(checkpoint['D'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        else:
            if isinstance(self.generator, torch.nn.DataParallel):
                self.generator.module.load_state_dict(checkpoint['G'])
            else:
                self.generator.load_state_dict(checkpoint['G'])
        return checkpoint['epoch']
        
    def forward(self):
        self.output = self.generator(self.content,self.style)                

    def backward_G(self):
        # GAN Loss
        self.loss_G_GAN = self.discriminator(self.output, self.style, mode='generator')['Sum'].mean()
        
        # Perceptual Loss
        zero_loss = torch.zeros(1).to(self.device)
        self.loss_G_content = zero_loss if self.lambda_content <= 0 else \
                self.criterion_content(self.output, self.content).mean()
                
        self.loss_G_style = zero_loss if self.lambda_style <= 0 else \
                self.criterion_style(self.output, self.style).mean()
        # Total Loss
        self.loss_G = self.lambda_GAN * self.loss_G_GAN + self.lambda_content * self.loss_G_content +\
                      self.lambda_style * self.loss_G_style
           
        self.loss_G.backward()
        
    def backward_D(self):
        self.loss_D_GAN = self.discriminator(self.output, self.style, mode='discriminator')['Sum'].mean() +\
                          self.discriminator(self.content, self.style, mode='discriminator')['Sum'].mean()
        self.loss_D = self.lambda_GAN * self.loss_D_GAN
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        # Optimize the Generator
        for param in self.discriminator.parameters():
                    param.requires_grad = False        
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Optimize the Discriminator
        for param in self.discriminator.parameters():
                    param.requires_grad = True
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
            
    def test(self, input):
        self.content = input['content'].to(self.device)
        self.style = input['style'].to(self.device)
        with torch.no_grad():
            self.forward()
        return self.output
    
    def save_img(self,path, name):
        imgA = Image.fromarray(tensor2im(self.content))
        imgB = Image.fromarray(tensor2im(self.style))
        imgA2B = Image.fromarray(tensor2im(self.output))
        
        imgA.save(os.path.join(path,str(name)+'_A.png'))
        imgB.save(os.path.join(path,str(name)+'_B.png'))
        imgA2B.save(os.path.join(path,str(name)+'_A2B.png'))
        return tensor2im(self.content), tensor2im(self.style), tensor2im(self.output)
        
def tensor2im(input_image, imtype=np.uint8):
    sflag = False
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.ndim == 2:
        sflag = True
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.round((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0)
    if sflag:
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)