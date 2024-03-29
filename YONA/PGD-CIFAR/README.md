## Pytorch Adversarial Training on CIFAR-10

* This repository provides simple PyTorch implementations for adversarial training methods on CIFAR-10.
* This repository shows accuracies that are similar to the accuracies in the original papers.
* If you have questions about this repository, please send an e-mail to me (dongbinna@postech.ac.kr) or make an issue.

### Experiment Settings

* The basic experiment setting used in this repository follows the setting used in [Madry Laboratory](https://github.com/MadryLab/cifar10_challenge).
* Dataset: CIFAR-10 (10 classes)
* Attack method: PGD attack
  1) Epsilon size: 0.0314 for <b>L-infinity bound</b>
  2) Epsilon size: 0.25 (for attack) or 0.5 (for training) for <b>L2 bound</b>
* Training batch size: 128
* Weight decay: 0.0002
* Momentum: 0.9
* Learning rate adjustment
  1) 0.1 for epoch [0, 100)
  2) 0.01 for epoch [100, 150)
  3) 0.001 for epoch [150, 200)
* The ResNet-18 architecture used in this repository is smaller than Madry Laboratory, but its performance is similar.

### Training Methods

####  PGD Adversarial Training

* This defense method was proposed by Aleksander Madry in [ICLR 2018](https://arxiv.org/pdf/1706.06083.pdf).
<pre>
python3 pgd_adversarial_training.py
</pre>