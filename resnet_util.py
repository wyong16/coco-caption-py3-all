import torch
import torch.nn as nn
import torchvision.models as models

class ResnetBackbone(nn.Module):
    def __init__(self):
        super(ResnetBackbone, self).__init__()
        # ResNet-101 backend
        resnet = models.resnet101()
        state_dict = torch.load("/kaggle/input/resnet101/resnet101.pth")
        resnet.load_state_dict(state_dict)
        modules = list(resnet.children())[:-2]  # delete the last fc layer and avg pool.
        self.resnet_conv = nn.Sequential(*modules)  # last conv feature
    def forward(self, image):
        conv_feat = self.resnet_conv(image)
        b, c, h, w = conv_feat.size()
        return conv_feat.view(b, c, -1).transpose(1,2)
