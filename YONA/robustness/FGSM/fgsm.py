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


def get_num_correct(out, labels):  # get accuracy
    return out.argmax(dim=1).eq(labels).sum().item()

# read the pre-trained model
model = torch.load('vgg19.pth',map_location='cpu')
model=model.to(device)
model.eval()

test_accuracy = 0
with torch.no_grad():
  for (images,labels) in test_loader:
      images,labels = images.to(device),labels.to(device)
      outs = model(images)
      test_accuracy += get_num_correct(outs,labels)
test_accuracy /= n_test


def fgsm_attack(image, epsilon, data_grad):
    if epsilon == 0:
        return image
    else:
        image = unnormalize(image)
        pertubed_image = image + epsilon * data_grad.sign()
        pertubed_image = torch.clamp(pertubed_image, 0, 1)
        pertubed_image = transforms.Normalize(mean=norm_mean, std=norm_std)(pertubed_image)
        return pertubed_image.float()


def fgsm_test(model, data_loader, epsilon, n_examples):
  '''
  input:
    data_loader: data set, batch size = 1
    epsilon: parameter to perform fgsm attack
  return:
    final_acc: accuracy of the model on classifying adversarial examples created based on datas
    adv_examples: n_examples examples of successed adversrial examples
  '''
  print('Epsilon:', epsilon)
  print('-' * 10)
  correct = 0
  adv_examples = []
  # Loop over all examples in data set, data shape: (C, H, W)
  for i, (data, target) in enumerate(data_loader):

      if i>0 and i%1000 == 0:
        current_acc = correct/i
        print(f'Test Accuracy = {current_acc:.4f} [{i:>5d} / {len(data_loader):>5d}]')

      # Send the data and label to the device
      data, target = data.to(device), target.to(device)

      # Set requires_grad attribute of tensor. Important for Attack
      data.requires_grad = True

      # Forward pass the data through the model
      output = model(data)
      init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
      # If the initial prediction is wrong, dont bother attacking, just move on
      if init_pred.item() != target.item():
          continue

      # Calculate the loss
      loss = F.nll_loss(output, target)

      # Zero all existing gradients
      model.zero_grad()

      # Calculate gradients of model in backward pass
      loss.backward()

      # Collect datagrad
      data_grad = data.grad.data

      # Call FGSM Attack
      perturbed_data = fgsm_attack(data, epsilon, data_grad)

      # Re-classify the perturbed image
      output = model(perturbed_data)

      # Check for success
      final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
      if final_pred.item() == target.item():
          correct += 1
          # Special case for saving 0 epsilon examples
          if (epsilon == 0) and (len(adv_examples) < n_examples):
              adv_ex = perturbed_data.squeeze().detach().cpu()
              ori_ex = data.squeeze().detach().cpu()
              adv_examples.append((init_pred.item(), final_pred.item(), adv_ex, ori_ex))
      else:
          # Save some adv examples for visualization later
          if len(adv_examples) < n_examples:
              adv_ex = perturbed_data.squeeze().detach().cpu()
              ori_ex = data.squeeze().detach().cpu()
              adv_examples.append((init_pred.item(), final_pred.item(), adv_ex, ori_ex))

  # Calculate final accuracy for this epsilon
  final_acc = correct / float(len(data_loader))
  print("Epsilon: {}\tTest Accuracy = {} / {} = {} \n".format(epsilon, correct, len(data_loader), final_acc))

  # Return the accuracy and an adversarial example
  return final_acc, adv_examples


epsilons = [0.1,0.2,0.3]
n_examples = 5
examples = []
accuracies = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = fgsm_test(model, test_loader_1, eps, n_examples)
    accuracies.append(acc)
    examples.append(ex)