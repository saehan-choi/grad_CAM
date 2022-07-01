from pyexpat import model
import torch.nn as nn
import torch
import timm

class CFG:
    clsarr = ['nomask', 'mask', 'wrong', 'blind']
    classes = 4

class efficientnet_b0:
    def __init__(self):
        super(efficientnet_b0, self).__init__()
        
        self.model = timm.create_model('efficientnet_b0')
        self.global_pool = self.model.global_pool
        self.classifier = nn.Linear(1280, CFG.classes)
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        
        self.gradient = None
        
    def forward(self, x):
        x = self.model(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        # input:256 -> ([1, 1280, 8, 8])
        
        
        
        
        
efficientnet_b0()