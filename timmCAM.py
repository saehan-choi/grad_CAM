from pyexpat import model
import torch.nn as nn
import torch
import timm

class CFG:
    clsarr = ['nomask', 'mask', 'wrong', 'blind']
    classes = 4

class efficientnet_b0(nn.Module):
    def __init__(self):
        super(efficientnet_b0, self).__init__()
        
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.global_pool = self.model.global_pool
        self.classifier = nn.Linear(1280, CFG.classes)
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        
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
        # input:256 -> ([1, 1280, 8, 8])
        
modelE = efficientnet_b0()
modelE.eval()
randn = torch.randn(1,3,256,256)

pred = modelE(randn)

argmax = pred.argmax().item()
print(f'{CFG.clsarr[argmax]}로 예측하였습니다.')

print(pred.size())
pred[:, argmax].backward()
gradients = modelE.get_activations_gradient()
