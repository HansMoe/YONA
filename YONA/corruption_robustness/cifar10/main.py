import argparse
import os
import random
import shutil
import time
import warnings
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import autoaugment
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from models.resnet import resnet18
from models.xception import xception
from models.preactresnet import preactresnet18
from models.densenet import densenet121
from models.resnext import resnext50
from models.wideresidual import wideresnet
from torchvision.transforms import autoaugment
from transformers import AutoImageProcessor, SwinForImageClassification
import cv2
# from augmentations import YONARandAugment, RandAugment, MaskHalf
from randomAug import Rand_Augment

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--dataset', dest='dataset', default='cifar100', type=str,
                    help='dataset (options: cifar10, cifar100)')
parser.add_argument('--salmix', dest='salmix', action='store_true',
                    help='apply saliencymix, performed with YOCO')
parser.add_argument('--beta', default=1, type=float,
                    help='hyperparameter beta')
parser.add_argument('--salmix_prob', default=0.5, type=float,
                    help='SaliencyMix probability')
parser.add_argument('--aug_method', default='HFlip', type=str,
                    help='use what data augmentation method to combine')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', type=str,
                    help='models, choose from resnet18, xception, densenet121, resnext50, wideresnet, vit')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-m', '--mixup', dest='mixup', action='store_true',
                    help='apply mixup, performed with YOCO')
parser.add_argument('--cutmix', dest='cutmix', action='store_true',
                    help='apply cutmix, performed with YOCO')
parser.add_argument('--yona', dest='yona', action='store_true',
                    help='apply yona')
parser.add_argument('--aug', dest='aug', action='store_true',
                    help='use data augmentation or not')
parser.add_argument('--maintain_rate', default=0.5, type=float,
                    help='the ratio of the mask image')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()
    print(args.dataset)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


# define random_replace function
def random_replace(img):
    img_tensor = torch.tensor(img)  

    mask = torch.rand(img_tensor.size()) < 0.1  
    r = torch.rand(img_tensor.size())
    img_tensor[mask] = r[mask]  

    return img_tensor  


# define add_gaussian_noise function
def add_gaussian_noise(img, mean=0, std=0.1):
    noise = torch.randn(img.size()) * std + mean
    noisy_img = img + noise
    return noisy_img


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        if args.dataset == "cifar10":
            num_classes = 10
        else:
            num_classes = 100
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch == "swin":
            model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            model.classifier = nn.Linear(model.swin.num_features, num_classes)
        else:
            model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch == 'resnet18':
            model = preactresnet18().cuda()
        if args.arch == 'densenet121':
            model = densenet121().cuda()
        if args.arch == 'xception':
            model = xception().cuda()
        if args.arch == 'resnext50':
            model = resnext50().cuda()
        if args.arch == 'wideresnet':
            model = wideresnet(depth=28).cuda()
        if args.arch == "vit":
            from models.vit_small import ViT
            # ViT for cifar10
            model = ViT(
                image_size=32,
                patch_size=4,
                num_classes=10,
                dim=512,
                depth=6,
                heads=8,
                mlp_dim=512,
                dropout=0.1,
                emb_dropout=0.1
            ).cuda()
        else:
            model = resnet18().cuda()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.PILToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: random_replace(x)), # use random_replace
            transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0, std=0.1)),  # use gaussian_noise
            normalize
        ])

        if args.aug_method == "randAug":
            if args.dataset == 'cifar10':
                transform_train.transforms.insert(0, YONARandAugment(3, 9))
            else:
                transform_train.transforms.insert(0, YONARandAugment(2, 14))
        elif args.aug_method == "autoAug":
            transform_train.transforms.insert(0, MaskHalf)
            if args.dataset == 'cifar10':
                transform_train.transforms.insert(1, transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
            else:
                transform_train.transforms.insert(1, transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
    print(best_acc1)


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
        img = img * mask

        return img


def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


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
            img = img * mask + offset
        else:
            img = img * mask

        return img


def train(train_loader, model, criterion, optimizer, epoch, args):
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
    else:
        transform = transforms.Compose([
        ])

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    to_pil = transforms.ToPILImage()
    # image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # inputs = image_processor(images, return_tensors="pt")

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # apply YONA
        if not args.mixup and not args.cutmix and not args.salmix and args.aug_method != "randAug" and args.aug_method != "autoAug":
            if args.yona:
                images = Mask_Half(images, 32, 32)
            images = transform(images)
        images = normalize((images / 255)).to(dtype=torch.float32)

        # perform mixup in yona way
        if args.mixup:
            images = Mask_Half(images, 32, 32)
            id = torch.randperm(images.size()[0]).cuda()
            images, lam, target_a, target_b = mixup(images, id, target)
            output = model(images.contiguous())
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        elif args.cutmix:
            images = Mask_Half(images, 32, 32)
            # generate mixed sample
            lam = np.random.beta(0.5, 0.5)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            # compute output
            output = model(images)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        elif args.salmix:
            images = Mask_Half(images, 32, 32)
            r = np.random.rand(1)
            if args.beta > 0 and r < args.salmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                labels_a = target
                labels_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = saliency_bbox(images[rand_index[0]], lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                output = model(images.contiguous())
                loss = criterion(output, labels_a) * lam + criterion(output, labels_b) * (1. - lam)
            else:
                output = model(images.contiguous())
                loss = criterion(output, target)
        elif args.arch == "swin":
            output = model(**inputs).logits
            loss = criterion(output, target)
        else:
            output = model(images.contiguous())
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def Mask_Half(x, h, w):
    if torch.rand(1) > 0.5:
        img1 = x[:, :, :, 0:int(w / 2)]
        img2 = x[:, :, :, int(w / 2):w]
        if torch.rand(1) > 0.5:
            zero_img = torch.zeros(img1.size(0), img1.size(1), img1.size(2), img1.size(3))
            images = torch.cat((zero_img, img2), dim=3)
        else:
            zero_img = torch.zeros(img2.size(0), img2.size(1), img2.size(2), img2.size(3))
            images = torch.cat((img1, zero_img), dim=3)
    else:
        img1 = x[:, :, 0:int(h / 2), :]
        img2 = x[:, :, int(h / 2):h, :]
        if torch.rand(1) > 0.5:
            zero_img = torch.zeros(img1.size(0), img1.size(1), img1.size(2), img1.size(3))
            images = torch.cat((zero_img, img2), dim=2)
        else:
            zero_img = torch.zeros(img2.size(0), img2.size(1), img2.size(2), img2.size(3))
            images = torch.cat((img1, zero_img), dim=2)

    return images


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup(input, rand_index, target):
    lam = np.random.beta(1, 1)
    target_b = target[rand_index]
    input = lam * input + (1 - lam) * input[rand_index, :]
    return input, lam, target, target_b


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        progress.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                       .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches=0, meters=None, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.file = open(os.path.join('log.txt'), 'a')

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        info = '\t'.join(entries)
        print(info)
        self.file.write(info + '\n')
        sys.stdout.flush()
        self.file.flush()

    def write(self, info):
        print(info)
        self.file.write(info + '\n')
        sys.stdout.flush()
        self.file.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
