'''
modified from: https://github.com/pacifinapacific/StyleGAN_LatentEditor/blob/master/perceptual_model.py
'''


import torch 
from torchvision import models 
import torch.nn as nn 


class VGG16_for_Perceptual(torch.nn.Module):
    def __init__(self,requires_grad=False,n_layers=[2,4,14,21]):
        super(VGG16_for_Perceptual,self).__init__()
        vgg_pretrained_features=models.vgg16(pretrained=True).features 

        self.slice0=torch.nn.Sequential()
        self.slice1=torch.nn.Sequential()
        self.slice2=torch.nn.Sequential()
        self.slice3=torch.nn.Sequential()

        for x in range(n_layers[0]):#relu1_1
            self.slice0.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[0],n_layers[1]): #relu1_2
            self.slice1.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[1],n_layers[2]): #relu3_2
            self.slice2.add_module(str(x),vgg_pretrained_features[x])

        for x in range(n_layers[2],n_layers[3]):#relu4_2
            self.slice3.add_module(str(x),vgg_pretrained_features[x])

        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False
        

    
    def forward(self,x):
        h0=self.slice0(x)    #torch.Size([1, 64, 256, 256])
        h1=self.slice1(h0)   #torch.Size([1, 64, 256, 256])
        h2=self.slice2(h1)   #torch.Size([1, 256, 64, 64])
        h3=self.slice3(h2)   #torch.Size([1, 512, 32, 32])

        return h0,h1,h2,h3


