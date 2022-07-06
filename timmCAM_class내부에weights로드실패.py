
from sre_constants import CH_LOCALE
from turtle import forward
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import torch.nn as nn
import torch
import timm

class CFG:
    clsarr = ['nomask', 'mask', 'wrong', 'blind']
    classes = 4
    weights_path = './weights/256_efficientnet_b0_3epoch.pt'
    
    device = 'cuda'
    
    img_resize = (256,256)
    
    transformed = A.Compose([A.Resize(img_resize[0], img_resize[1]),
                            ToTensorV2()
                            ])
    
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = timm.create_model('efficientnet_b0', num_classes=CFG.classes)
#         self.model.load_state_dict = torch.load(CFG.weights_path)
        
#     def forward(self, x):
#         return self.model(x)
    
# model = Model()
# model.load_state_dict(torch.load(CFG.weights_path))
# model.to(CFG.device)

class efficient(nn.Module):
    def __init__(self):
        super(efficient, self).__init__()

        self.model = timm.create_model('efficientnet_b0', num_classes=CFG.classes)
        print(self.model)
        # self.model.load_state_dict(torch.load(CFG.weights_path))
        # self.model.load_state_dict(torch.load(CFG.weights_path), strict=False)
        # print(self.model.state_dict())
        
        # print(self.model.load_state_dict(torch.load(CFG.weights_path)))
        # self.model.state_dict() = torch.load(CFG.weights_path)

        self.global_pool = self.model.global_pool
        self.classifier = self.model.classifier

        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        # print(self.model)
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

def img_transform(img):
    transformed = CFG.transformed(image=img)
    transformed_img = transformed["image"].unsqueeze(0).float().to(CFG.device)
    return transformed_img

# print(model)

randn = torch.ones(1,3,256,256)


model = efficient()


model.eval()

output = model(randn)
