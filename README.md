# style_transfer_real2monet


![Architecture](/image/architecture.PNG)
|Target|![](/image/1_target.png) |![](/image/2_target.png)|![](/image/3_target.png) 
|:---:|:---:|:---:|:---:|
|Predict|![](/image/1_predict.png) |![](/image/2_predict.png)|![](/image/3_predict.png) 

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



