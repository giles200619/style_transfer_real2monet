# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monet_dataloader import MonetDataset
from options.train_options import TrainOptions
from models.monet_model import StyleTransfer


if __name__ == '__main__':
    opt = TrainOptions().parse()
    
    train_dataset = MonetDataset(opt.data_dir,train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    
    model = StyleTransfer(opt)
    if opt.continue_train:
        opt.start_epoch = model.load(opt.load_epoch,opt.checkpoints_dir)
        print("continue training from epoch:",opt.start_epoch)
        
    # Tensorboard summary writer
    writer = SummaryWriter(f'runs/{opt.name}')
    
    total_steps = 0
    for epoch in range(opt.start_epoch, opt.num_epochs):
        epoch_iter = 0
        total_steps -= total_steps%1000
        for ii, batch in enumerate(train_dataloader):
            
            batch_size = batch['content'].shape[0]
            total_steps += batch_size
            epoch_iter += batch_size
            
            # Optimize the model
            model.set_input(batch)
            model.optimize_parameters()
            
            if total_steps % opt.print_freq == 0:
                writer.add_scalar('generator_loss', model.loss_G.item(), total_steps)
                writer.add_scalar('generator_GAN_loss', model.loss_G_GAN.item(), total_steps)
                writer.add_scalar('generator_content_loss', model.loss_G_content.item(), total_steps)
                writer.add_scalar('generator_style_loss', model.loss_G_style.item(), total_steps)
                writer.add_scalar('discriminator_loss', model.loss_D.item(), total_steps)
                writer.add_scalar('discriminator_GAN_loss', model.loss_D_GAN.item(), total_steps)
                writer.add_scalar('total_loss', model.loss_G.item()+model.loss_D.item(), total_steps)
                print('Epoch:',epoch,'iteration:',epoch_iter,
                      'loss:', "{:.5f}".format(model.loss_G.item()+model.loss_D_GAN.item()),
                      'G:','%.5f' % model.loss_G.item(),
                      'D:','%.5f' % model.loss_D.item(),
                      'G_GAN:', '%.5f' % model.loss_G_GAN.item(),
                      'G_content:','%.5f' % model.loss_G_content.item(),
                      'G_style:','%.5f' % model.loss_G_style.item(),
                      'D_GAN:','%.5f' % model.loss_D_GAN.item())
                
            if total_steps % opt.save_img_freq ==0:
                imgA, imgB, imgA2B = model.save_img(opt.train_result_dir, total_steps)
                fig, axs = plt.subplots(1,3)
                axs[0].imshow(imgA)
                axs[0].axis('off')
                axs[0].title.set_text('Img A')
                axs[1].imshow(imgB)
                axs[1].axis('off')
                axs[1].title.set_text('Img B')
                axs[2].imshow(imgA2B)
                axs[2].axis('off')
                axs[2].title.set_text('Img A2B')
                writer.add_figure('training results', fig, total_steps)
                
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch {}, total_steps {})'.format(epoch, total_steps))
                model.save(epoch, latest=True)
        # Save the model to checkpoint directory
        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch: {}'.format(epoch))
            model.save(epoch, latest=True)
            model.save(epoch, latest=False)
            
    print('Finished training!')
            
