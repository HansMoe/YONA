# You Only Need Half: Boosting Data Augmentation by Using Partial Content（YONA）
Code release for [You Only Need Half: Boosting Data Augmentation by Using Partial Content](https://arxiv.org/abs/2405.02830)

##  The illustrations of YONA operations
YONA methodically bisects an image into two equal segments along either the vertical or horizontal axis. Subsequently, it applies data augmentation to one segment, masks the other with noise, and finally, reassembles the two altered segments into a cohesive image.
![YONA示意图](https://github.com/HansMoe/YONA/blob/main/YONA.png)

## Training

### CIFAR10
First
```
cd YONA/image_classification/cifar10
```
Run with YONA on ResNet-18 with HFlip
```
python main.py --arch resnet18 --aug_method HFlip
```
Run with YONA on ResNet-18 with VFlip
```
python main.py --arch resnet18 --aug_method VFlip
```
Run with YONA on ResNet-18 with Jitter
```
python main.py --arch resnet18 --aug_method Jitter
```
Run with YONA on ResNet-18 with Erasing
```
python main.py --arch resnet18 --aug_method Erasing
```
Run with YONA on ResNet-18 with Cutout
```
python main.py --arch resnet18 --aug_method Cutout
```
Run with YONA on ResNet-18 with Grid
```
python main.py --arch resnet18 --aug_method Grid
```
Run with YONA on ResNet-18 with autoAug
```
python main.py --arch resnet18 --aug_method autoAug
```
Run with YONA on ResNet-18 with randomAug
```
python main.py --arch resnet18 --aug_method randomAug
```

## Citation
If you use this code for your research, please consider citing:
```
@article{hu2024you,
  title={You Only Need Half: Boosting Data Augmentation by Using Partial Content},
  author={Hu, Juntao and Wu, Yuan},
  journal={arXiv preprint arXiv:2405.02830},
  year={2024}
}
```
