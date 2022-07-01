import torch.nn as nn
import torch
import timm


class efficientnet_b0:
    def __init__(self):
        super(efficientnet_b0, self).__init__()
        
        model = timm.create_model('efficientnet_b0')
        model.global_pool = nn.Identity()
        model.classifier = nn.Identity()

        randn = torch.randn(1,3,256,256)
        print(model(randn).size())
        
        
        
        
efficientnet_b0()