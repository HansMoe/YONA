import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--aug_method', default='HFlip', type=str,
                    help='use what data augmentation method to combine')

args = parser.parse_args()

batch_size = 64
input_size = 32
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

train_data=dset.CIFAR10(root='../data/', download=True,train=True,transform=transform)
test_data=dset.CIFAR10(root='../data/', download=True, train=False,transform=transform)
train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)
test_loader_1=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=True) # used for adversarial attack
n_train=len(train_data)
n_test=len(test_data)
classes = train_data.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device))

def unnormalize(img, mean = np.array(norm_mean), std = np.array(norm_std)):
  '''
   unnormalize the image that has been normalized with mean and std
  '''
  inverse_mean = - mean/std
  inverse_std = 1/std
  img = transforms.Normalize(mean=-mean/std, std=1/std)(img)
  return img

def normalize(img, mean = np.array(norm_mean), std = np.array(norm_std)):
  return transforms.Normalize(mean = norm_mean, std = norm_std)(img)


def get_num_correct(out, labels):  #求准确率
    return out.argmax(dim=1).eq(labels).sum().item()


def Mask_Half(x, h, w):
    if torch.rand(1) > 0.5:
        img1 = x[:, :, :, 0:int(w / 2)]
        img2 = x[:, :, :, int(w / 2):w]
        if torch.rand(1) > 0.5:
            zero_img = torch.zeros(img1.size(0), img1.size(1), img1.size(2), img1.size(3))
            images = torch.cat((zero_img.to('cuda'), img2), dim=3)
        else:
            zero_img = torch.zeros(img2.size(0), img2.size(1), img2.size(2), img2.size(3))
            images = torch.cat((img1, zero_img.to('cuda')), dim=3)
    else:
        img1 = x[:, :, 0:int(h / 2), :]
        img2 = x[:, :, int(h / 2):h, :]
        if torch.rand(1) > 0.5:
            zero_img = torch.zeros(img1.size(0), img1.size(1), img1.size(2), img1.size(3))
            images = torch.cat((zero_img.to('cuda'), img2), dim=2)
        else:
            zero_img = torch.zeros(img2.size(0), img2.size(1), img2.size(2), img2.size(3))
            images = torch.cat((img1, zero_img.to('cuda')), dim=2)

    return images

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=1, length=0):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)  # Random location
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)  # The clipping area can be out of the picture
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask.to('cuda')

        return img

from PIL import Image

class Grid(object):
    def __init__(self, use_h, use_w, rotate=1, offset=True, ratio=0.005, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(2)
        w = img.size(3)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        # d = self.d
        #        self.l = int(d*self.ratio+0.5)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask.to('cuda') + offset.to('cuda')
        else:
            img = img * mask.to('cuda')

        return img


model=models.vgg19(pretrained=True) #这里采用VGG19模型
model.classifier[-1]=nn.Linear(4096, len(classes)) # 根据class个数修改最后一层的输出

model=model.to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

epoch_num = 20
best_acc = 0
train_loss = []
test_loss = []
test_accuracy = []
since = time.time()

if args.aug_method == "HFlip":
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])
elif args.aug_method == "VFlip":
    transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
    ])
elif args.aug_method == "Jitter":
    transform = transforms.Compose([
        transforms.ColorJitter(),
    ])
elif args.aug_method == "Erasing":
    transform = transforms.Compose([
        transforms.RandomErasing(),
    ])
elif args.aug_method == "Blur":
    transform = transforms.Compose([
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    ])
elif args.aug_method == "Cutout":
    transform = transforms.Compose([
        Cutout(n_holes=5, length=1),
    ])
elif args.aug_method == "autoAug":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.autoaugment.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
    ])
elif args.aug_method == "randomAug":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        autoaugment.RandAugment(),
        transforms.ToTensor(),
    ])
elif args.aug_method == "Grid":
    transform = transforms.Compose([
        Grid(use_h=32, use_w=32),
    ])

for epoch in range(epoch_num):
    print('Epoch {}/{}'.format(epoch + 1, epoch_num))
    print('-' * 10)
    train_loss_tmp = 0
    test_loss_tmp = 0
    test_accuracy_tmp = 0

    model.train()
    for batch, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        images = Mask_Half(images, 32, 32)
        images = transform(images)
        outs = model(images)
        loss = F.cross_entropy(outs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_tmp += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{n_train:>5d}]")
    train_loss_tmp /= n_train

    model.eval()
    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)
            outs = model(images)
            test_loss_tmp += F.cross_entropy(outs, labels).item()
            test_accuracy_tmp += get_num_correct(outs, labels)
    test_loss_tmp /= n_test
    test_accuracy_tmp /= n_test

    train_loss.append(train_loss_tmp)
    test_loss.append(test_loss_tmp)
    test_accuracy.append(test_accuracy_tmp)

    if test_accuracy_tmp > best_acc:
        torch.save(model, 'vgg19.pth')
        best_acc = test_accuracy_tmp
    print(f"Test Error: \n Accuracy: {(100 * test_accuracy_tmp):>0.1f}%, Avg loss: {test_loss_tmp:>8f} \n")
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

