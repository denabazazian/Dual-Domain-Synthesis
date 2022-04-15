'''
modified from : https://github.com/bryandlee/repurpose-gan
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
from matplotlib import pyplot as plt

import argparse
import numpy as np
import time
import os
import random
import copy
import cv2
import glob
import math

from model_repurpose import FewShotCNN
from labeller import Labeller
from utils_repurpose import tensor2image, imshow, horizontal_concat, imsave
from stylegan2  import Generator

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--generator_dir", type=str, help='path to the generator model')
parser.add_argument("--segmentation_dir", type=str, help='path to the save the segmentation model')
parser.add_argument("--part_key", type=str, help='keyword to the segmentation part')
parser.add_argument("--image_size", type=int, default=256 , help='image size')
parser.add_argument("--n_samples", type=int, default=1 , help='number of training samples')
parser.add_argument("--imshow_size", type=int, default=3 , help='image show size')
parser.add_argument("--latent_dim", type=int, default=512 , help='latent dimension')
parser.add_argument("--truncation", type=float, default=0.7 , help='truncation')
parser.add_argument("--n_test", type=int, default=3 , help='number of test images')

################################################################################
args = parser.parse_args()

generator_path = args.generator_dir
segmentation_path = args.segmentation_dir
segmentation_part = args.part_key
image_size = args.image_size 
n_samples = args.n_samples
imshow_size = args.imshow_size
latent_dim =  args.latent_dim
truncation =  args.truncation
n_test=  args.n_test 

############################################
save_path = segmentation_path+"/"+segmentation_part

if not os.path.exists(save_path):
    os.makedirs(save_path)
#############################################

device = 'cuda:0'

generator = Generator(image_size, latent_dim, 8)
generator_ckpt = torch.load(generator_path, map_location='cpu')
generator.load_state_dict(generator_ckpt["g_ema"], strict=False)
generator.eval().to(device)
print(f'[StyleGAN2 generator loaded] {generator_path}\n')

with torch.no_grad():
    trunc_mean = generator.mean_latent(4096).detach().clone()
    latent = generator.get_latent(torch.randn(n_samples, latent_dim, device=device))
    imgs_gen, features = generator([latent],
                                   truncation=truncation,
                                   truncation_latent=trunc_mean,
                                   input_is_latent=True,
                                   randomize_noise=True)
    torch.cuda.empty_cache()

print("sample images:")
imshow(tensor2image(horizontal_concat(imgs_gen)), size=imshow_size*n_samples)


classes = ['background','eyes']


print("imgs_gen shape from pytorch is {}".format(imgs_gen.shape))        
labeller = Labeller(imgs_gen.clamp_(-1., 1.).detach().permute(0,2,3,1).cpu().numpy() * 0.5 + 0.5, classes)

labels = labeller.get_labels()

imshow(tensor2image(horizontal_concat(imgs_gen)), size=imshow_size*n_samples)
imshow(np.concatenate([labeller.get_visualized_label(l) for l in labels], axis=1), size=imshow_size*n_samples)

@torch.no_grad()
def concat_features(features):
    h = max([f.shape[-2] for f in features])
    w = max([f.shape[-1] for f in features])
    return torch.cat([torch.nn.functional.interpolate(f, (h,w), mode='nearest') for f in features], dim=1)

data = dict(
    latents=latent.cpu(),
    features=concat_features(features).cpu(),
    labels=torch.tensor(labels).long(),
)
print(f'Dataset for {n_samples}-Shot Training is Prepared')

net = FewShotCNN(data['features'].shape[1], len(classes), size='S')

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

net.train().to(device)
start_time = time.time()

for epoch in range(1, 100+1):
    sample_order = list(range(n_samples))
    random.shuffle(sample_order)

    for idx in sample_order:
        
        sample = data['features'][idx].unsqueeze(0).to(device) # torch.Size([1, 5376, 256, 256])
        label = data['labels'][idx].unsqueeze(0).to(device) #torch.Size([1, 256, 256])

        out = net(sample)

        loss = F.cross_entropy(out, label, reduction='mean')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 50 == 0:
        print(f'{epoch:5}-th epoch | loss: {loss.item():6.4f} | time: {time.time()-start_time:6.1f}sec')

    scheduler.step()
print('Training is Done!')

torch.save(net.state_dict(), save_path+"/"+segmentation_part+".pt")
print('The model is saved.')

#############################################
print('Start Testing')

device = 'cpu' 
generator.eval().to(device) 

print('loading the model')
net = FewShotCNN(data['features'].shape[1], len(classes), size='S')
net.load_state_dict(torch.load(save_path+"/"+segmentation_part+".pt"))

net.eval().to('cpu')
print('model is loaded')

with torch.no_grad():
    latent = generator.get_latent(torch.randn(n_test, latent_dim, device=device))
    imgs_gen, features = generator([latent],
                                   truncation=truncation,
                                   truncation_latent=trunc_mean.to('cpu'), 
                                   input_is_latent=True,
                                   randomize_noise=True)
    torch.cuda.empty_cache()
    
    out = net(concat_features(features))
    predictions = out.data.max(1)[1].cpu().numpy()

    predictions = np.concatenate([labeller.get_visualized_label(pred) for pred in predictions], axis=1)
    results = np.concatenate([tensor2image(horizontal_concat(imgs_gen)), predictions], axis=0)
    imshow(results, size=imshow_size*n_test)
