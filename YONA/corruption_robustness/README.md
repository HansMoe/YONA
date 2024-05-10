Code for corruption_robustness.
## Training
### CIFAR
First
```
cd YONA/corruption_robustness/cifar10
```
Run with YONA on HFlip
```
python main.py --aug_method HFlip
```
Run with YONA on VFlip
```
python main.py --aug_method VFlip
```
Run with YONA on Jitter
```
python main.py --aug_method Jitter
```
Run with YONA on Erasing
```
python main.py --aug_method Erasing
```
Run with YONA on Cutout
```
python main.py --aug_method Cutout
```
Run with YONA on Grid
```
python main.py --aug_method Grid
```
Run with YONA on autoAug
```
python main.py --aug_method autoAug
```
Run with YONA on randomAug
```
python main.py --aug_method randomAug
```
