Code for robustness under FGSM attack. 

## Training

### CIFAR
First
```
cd YONA/robustness/FGSM
```
Run the code for robustness under FGSM attack on HFlip
```
python main.py --aug_method HFlip
```
Run the code for robustness under FGSM attack on VFlip
```
python main.py --aug_method VFlip
```
Run the code for robustness under FGSM attack on Jitter
```
python main.py --aug_method Jitter
```
Run the code for robustness under FGSM attack on Erasing
```
python main.py --aug_method Erasing
```
Run the code for robustness under FGSM attack on Cutout
```
python main.py --aug_method Cutout
```
Run the code for robustness under FGSM attack on Grid
```
python main.py --aug_method Grid
```
Run the code for robustness under FGSM attack on randomAug
```
python main.py --aug_method randomAug
```
Run the code for robustness under FGSM attack on autoAug
```
python main.py --aug_method autoAug
```
