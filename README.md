# You Only Need Half: Boosting Data Augmentation using Partial Content(YONA)
YONA cuts one image into two equal pieces, in either the height dimension or the width dimension, performs data augmentation within one piece, masks out the other piece with noises, and concatenates the two transformed pieces back together.
![YONA](https://github.com/HansMoe/YONA/blob/main/YONA.png)

## Training
The provided code is an example of applying YONA to HFlip. Other augmentations are similar. 


### CIFAR
First
```
cd image_classification/cifar10
```
Run with YONA on ResNet-18
```
python main.py --arch resnet18 --aug_method HFlip
```



