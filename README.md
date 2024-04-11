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

For calibration on MoCo dataset, see [MoCo](https://github.com/facebookresearch/moco/tree/main/detection).

For calibration, see [PixMix](https://github.com/andyzoujm/pixmix).

For corruption robustness, see [Co-Mixup](https://github.com/snu-mllab/Co-Mixup).

For Adversarial robustness, see [ICLR2018](https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR).



