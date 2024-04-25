# You Only Need Half: Boosting Data Augmentation using Partial Content(YONA)
YONA cuts one image into two equal pieces, in either the height dimension or the width dimension, performs data augmentation within one piece, masks out the other piece with noises, and concatenates the two transformed pieces back together.
![YONA](https://github.com/HansMoe/YONA/blob/main/YONA.png)

## Training
The provided code is an example of applying YONA to HFlip. Other augmentations are similar. 


### CIFAR
First
```
cd classification/cifar10
```
Run with YONA on ResNet-18
```
python main.py --arch resnet18 --aug_method HFlip
```

For contrasitve learning, see [MoCo](https://github.com/facebookresearch/moco/tree/main/detection).

For calibration, see [PixMix](https://github.com/andyzoujm/pixmix).

For corruption robustness, see [Co-Mixup](https://github.com/snu-mllab/Co-Mixup).

For object detection task, see [Yolov7](https://github.com/WongKinYiu/yolov7)

For Adversarial robustness, see [ICLR2018](https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR).



