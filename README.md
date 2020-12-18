# style_transfer_real2monet
|Real|Real2Monet |Monet|
| --- | --- | --- |
|<img src="/images/10_A.png" width="150" />|<img src="/images/10_A2B.png" width="150" /> |<img src="/images/10_B.png" width="150" />|
|<img src="/images/21_A.png" width="150" />|<img src="/images/21_A2B.png" width="150" /> |<img src="/images/21_B.png" width="150" />|
|<img src="/images/32_A.png" width="150" />|<img src="/images/32_A2B.png" width="150" /> |<img src="/images/32_B.png" width="150" />|

Network architecture       |  Generator architecture
:-------------------------:|:-------------------------:
<img src="/images/network.PNG" width="250"> |  <img src="/images/generator.PNG" width="250">

Up Block for combining features and upsample      |  Down Block for style/content encoding
:-------------------------:|:-------------------------:
<img src="/images/upblock.PNG" width="200"> |  <img src="/images/upblock.PNG" width="200">

### Key Features:



## Getting Started
### Dependencies
* pytorch 1.5.1 
* torchvision 0.6.1 

## Data
The data is based on the Shapenet dataset naming convention: {model name}\_{azimuth/10}\_{elevation}.png

### Train
```
python train.py --train_dir [/folder/to/training/dataset] --batch_size [] --checkpoint [./checkpoint/0.pth]
```
### Test
```
python test.py --test_dir [/folder/to/testing/dataset] --checkpoint [./checkpoint/x.pth]
```



