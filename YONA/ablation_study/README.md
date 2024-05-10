Code for ablation study. 

## Training

### CIFAR
First
```
cd YONA/ablation_study/cifar10
```
Run with YONA on ResNet-18 with HFlip
```
python main.py --arch resnet18 --aug_method HFlip
```
Run with YONA on ResNet-18 with Jitter
```
python main.py --arch resnet18 --aug_method Jitter
```
Run with YONA on ResNet-18 with Erasing
```
python main.py --arch resnet18 --aug_method Erasing
```
Run with YONA on ResNet-18 with Grid
```
python main.py --arch resnet18 --aug_method Grid
```
