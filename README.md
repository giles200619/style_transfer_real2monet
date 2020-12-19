# style_transfer_real2monet
|Real|Real to Monet |Monet|
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
The provided pre-trained [weight](https://drive.google.com/file/d/1NHuOwIuEBvyPFQdGyG0ZFm5gfMvhIZuC/view?usp=sharing) is trained on this [dataset](https://www.kaggle.com/c/gan-getting-started/data).
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

