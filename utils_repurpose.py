# modified from : https://github.com/bryandlee/repurpose-gan
#---------------------------------------------------------------------------------


import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib.colors import hsv_to_rgb

import torch
import numpy as np
import time

import random
import copy
from PIL import Image as Img

from stylegan2 import Generator


def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    ##%matplotlib inline
    fig = plt.figure()
    #plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()
    #fig.savefig('/path/to/img_{}.png'.format(np.random.rand()))
    plt.close(fig)

def horizontal_concat(imgs):
    return torch.cat([img.unsqueeze(0) for img in imgs], 3)

def imsave(img,path,size,cmap='jet'):
    fig = plt.figure()
    #plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    #plt.show()
    fig.savefig(path+'img_{}.png'.format(np.random.rand()))
    plt.close(fig)
