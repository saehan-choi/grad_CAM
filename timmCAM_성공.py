from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A

from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch
import timm

import numpy as np

import cv2

import os

class CFG:
    clsarr = ['mask', 'nomask', 'wrong', 'blind']
    classes = 4
    
    img_path = './data/'
    write_path = './results/'
    weights_path = './weights/256_efficientnet_b0_3epoch.pt'
    
    device = 'cuda'
    
    img_resize = (256,256)
    
    transformed = A.Compose([A.Resize(img_resize[0], img_resize[1]),
                            ToTensorV2()
                            ])
    font =  cv2.FONT_HERSHEY_PLAIN
    color = (255, 255, 255)
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model('efficientnet_b0', num_classes=CFG.classes)
    
    def forward(self, x):
        return self.model(x)

class efficient(nn.Module):
    def __init__(self):
        super(efficient, self).__init__()

        self.model = model

        self.global_pool = self.model.model.global_pool
        self.classifier = self.model.model.classifier

        self.model.model.global_pool = nn.Identity()
        self.model.model.classifier = nn.Identity()
        self.gradient = None

    def activations_hook(self, grad):
        self.gradient = grad

    def get_activations_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.model(x)

    def forward(self, x):
        x = self.model(x)
        h = x.register_hook(self.activations_hook)
        # -> gradient 생성
        x = self.global_pool(x)
        x = self.classifier(x)

        return x

def img_transform(img_path):
    img = cv2.imread(img_path)
    transformed = CFG.transformed(image=img)
    transformed_img = transformed["image"].unsqueeze(0).float().to(CFG.device)
    return transformed_img


img_path = os.listdir(CFG.img_path)
labels = []
images = []

for i in img_path:
    all_path = CFG.img_path+i
    for j in os.listdir(all_path):
        labels.append(i)
        images.append(all_path+'/'+j)

model = Model()
model.load_state_dict(torch.load(CFG.weights_path))
model.to(CFG.device)
model.eval()

# 이게 진짜로 쓰이는 efficientNet B0 입니다.
modelE = efficient()

for i in range(len(labels)):
    datapath = images.pop(0)
    image = img_transform(datapath)
    original_label = labels.pop(0)

    output = modelE(image)
    
    pred = output.argmax().item()
    print(f'pred : {CFG.clsarr[pred]}')
    print(f'real : {original_label}')

    output[:, pred].backward()
    gradients = modelE.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
    
    
    # get the activations of the last convolutional layer
    activations = modelE.get_activations(image).detach()
    # weight the channels by corresponding gradients
    activations = activations * pooled_gradients
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = torch.clamp(heatmap, min=0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    # cuda to cpu. if not use it, error would occur
    heatmap = np.array(heatmap.cpu())

    heatmap = cv2.resize(heatmap, dsize=(CFG.img_resize[0], CFG.img_resize[1]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    heatmap = cv2.putText(heatmap, f'label:{original_label}', (5, 10), CFG.font, 1, CFG.color, 2, cv2.LINE_AA)
    heatmap = cv2.putText(heatmap, f"pred:{CFG.clsarr[pred]}", (5, 30), CFG.font, 1, CFG.color, 2, cv2.LINE_AA)
    
    image = image.squeeze().permute(1,2,0).cpu().numpy()
    
    superimposed_img = heatmap * 0.5 + image
    
    # print(superimposed_img)
    img_name = datapath.split('/')[3]
    
    if original_label!=CFG.clsarr[pred]:
        error_path = './error_results/'
        cv2.imwrite(error_path+img_name, superimposed_img)
    else:
        pass
    # 정상 이미지 저장
    # cv2.imwrite(CFG.write_path+img_name, superimposed_img)
    
    