import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

import cv2


# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder(root='./data/', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)



class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


# initialize the VGG model
vgg = VGG()

# set the evaluation mode
vgg.eval()

# get the image from the dataloader
img, _ = next(iter(dataloader))

# get the most likely prediction of the model
pred = vgg(img)

argmax = pred.argmax().item()
# get the gradient of the output with respect to the parameters of the model
print(f'{argmax}로 예측하였습니다.')
pred[:, argmax].backward()

# pull the gradients out of the model
gradients = vgg.get_activations_gradient()

# pool the gradients across the channels
# 이거 adaptive avg pooling (1,1) 으로 바꾸어도 똑같이 동작합니다.
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)

# get the activations of the last convolutional layer
activations = vgg.get_activations(img).detach()

# print(activations.size())
# [1,512,14,14]

# weight the channels by corresponding gradients
activations = activations * pooled_gradients

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = torch.clamp(heatmap, min=0)

# normalize the heatmap
heatmap /= torch.max(heatmap)
heatmap = np.array(heatmap)

img = cv2.imread('./elephant.jpg')

heatmap = cv2.resize(heatmap, dsize=(img.shape[1], img.shape[0]))
print(heatmap)
heatmap = np.uint8(255 * heatmap)


heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.7 + img

cv2.imwrite('./aa.jpg',superimposed_img)

