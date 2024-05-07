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
