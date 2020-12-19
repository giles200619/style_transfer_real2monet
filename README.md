# style_transfer_real2monet
A style transferring network that is conditioning on both the input content and style images. No cycle-consistency constraint is needed for training.
|Real|Real to Monet |Monet|
| --- | --- | --- |
|<img src="/images/10_A.png" width="150" />|<img src="/images/10_A2B.png" width="150" /> |<img src="/images/10_B.png" width="150" />|
|<img src="/images/21_A.png" width="150" />|<img src="/images/21_A2B.png" width="150" /> |<img src="/images/21_B.png" width="150" />|
|<img src="/images/32_A.png" width="150" />|<img src="/images/32_A2B.png" width="150" /> |<img src="/images/32_B.png" width="150" />|

Network architecture       |  Generator architecture
:-------------------------:|:-------------------------:
<img src="/images/network.PNG" width="340"> |  <img src="/images/generator.PNG" width="340">

Up Block for combining features and upsample      |  Down Block for style/content encoding
:-------------------------:|:-------------------------:
<img src="/images/upblock.PNG" width="200"> |  <img src="/images/upblock.PNG" width="200">

### Key Features:
* Content stream and style stream extract the features at different scale.
* AdaIN is used to combine the content feature and style feature at each level.
* Upsampling is done by interpolation followed by a convolutional layer.
* The score is set to 0 for the real image and 1 for the fake image when training the generator.
* The score is set to 0.1 for the real image and 0.9 for the fake image when training the discriminator with gan_mode = original.
* Gaussian noise is added to the image before passing into the discriminator.
* Both real and fake images are passed into the patch discriminator simultaneously.
* Content Loss is the MSE loss of the features extracted from one pre-trained VGG19 layer.
* Style Loss is the MSE loss between the features' Gram Matrix from several VGG19 layers.

## Getting Started
### Dependencies
* pytorch 1.5.1 
* torchvision 0.6.1 

## Data
To train A to B
```bash
└── data
    ├── trainA
    ├── tranB
    ├── testA
    └── testB
```

### Train
```
python train.py --data_dir [/folder/to/training/dataset] --batch_size [] 
```
To continue training:
```
python train.py --data_dir [/folder/to/training/dataset] --batch_size [] --continue_train
```
Please check options, train_options for more options.
### Test
The provided pre-trained [weight](https://drive.google.com/drive/folders/1uJy9VX7pSG-XPSI0RXZqbZENWYoAXvYF?usp=sharing) is trained on this [dataset](https://www.kaggle.com/c/gan-getting-started/data).
```
python test.py --data_dir [/folder/to/testing/dataset] --checkpoints_dir [] 
```

## More Results:
|Real|Real2Monet |Monet|
| --- | --- | --- |
|<img src="/images/10_A.png" width="150" />|<img src="/images/10_A2B.png" width="150" /> |<img src="/images/10_B.png" width="150" />|
|<img src="/images/12_A.png" width="150" />|<img src="/images/12_A2B.png" width="150" /> |<img src="/images/12_B.png" width="150" />|
|<img src="/images/13_A.png" width="150" />|<img src="/images/13_A2B.png" width="150" /> |<img src="/images/13_B.png" width="150" />|
|<img src="/images/20_A.png" width="150" />|<img src="/images/20_A2B.png" width="150" /> |<img src="/images/20_B.png" width="150" />|
|<img src="/images/21_A.png" width="150" />|<img src="/images/21_A2B.png" width="150" /> |<img src="/images/21_B.png" width="150" />|
|<img src="/images/22_A.png" width="150" />|<img src="/images/22_A2B.png" width="150" /> |<img src="/images/22_B.png" width="150" />|
|<img src="/images/30_A.png" width="150" />|<img src="/images/30_A2B.png" width="150" /> |<img src="/images/30_B.png" width="150" />|
|<img src="/images/31_A.png" width="150" />|<img src="/images/31_A2B.png" width="150" /> |<img src="/images/31_B.png" width="150" />|
|<img src="/images/32_A.png" width="150" />|<img src="/images/32_A2B.png" width="150" /> |<img src="/images/32_B.png" width="150" />|

## Acknowledgement
* Structure and part of the network code from: [CORN](https://github.com/nicolaihaeni/corn)
* [AdaIN implementation](https://github.com/naoto0804/pytorch-AdaIN)
* The design of the generator is inspired by [TSIT](https://arxiv.org/abs/2007.12072)
* Content loss and style loss are inspired by the [Pytorch Tutorials](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
