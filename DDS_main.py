import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time
import os
import random
import copy
import cv2
import glob
import math
import sys

from model_repurpose import FewShotCNN
from labeller import Labeller
from utils_repurpose import tensor2image, imshow, horizontal_concat, imsave
from perceptual_model import VGG16_for_Perceptual
from stylegan2 import Generator



################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--generator_domain1_dir", type=str, help='path to the domain1 generator model')
parser.add_argument("--generator_domain2_dir", type=str, help='path to the domain2 generator model')
parser.add_argument("--save_path_root", type=str, help='path to save the results')
parser.add_argument("--segmentation_dir", type=str, help='path to load the segmentation model')
parser.add_argument("--part_key", type=str, help='keyword to the segmentation part')
parser.add_argument("--image_size", type=int, default=256 , help='image size')
parser.add_argument("--n_samples", type=int, default=1 , help='number of training samples')
parser.add_argument("--imshow_size", type=int, default=3 , help='image show size')
parser.add_argument("--latent_dim", type=int, default=512 , help='latent dimension')
parser.add_argument("--truncation", type=float, default=0.7 , help='truncation')
parser.add_argument("--n_test", type=int, default=1 , help='number of test images')
parser.add_argument("--id_dir", type=str, default=1 , help='id to save files')
parser.add_argument("--sample_z_path", type=str, help='path to load the sample_z')
parser.add_argument("--save_iterations_path", type=str, help='path to save iterations')
parser.add_argument("--mask_guided_iterations",type=int,default=1002,help='number of the iterations')
parser.add_argument("--lr",type=float,default=0.01,help='learning rate')
parser.add_argument("--n_mean_latent",type=int,default=10000,help='n_mean_latent')
 
################################################################################

args = parser.parse_args()

generator_path = args.generator_domain1_dir
target_model_path = args.generator_domain2_dir
save_path_root = args.save_path_root
save_segmentation_model_path = args.segmentation_dir
segmentation_part = args.part_key
image_size = args.image_size 
n_samples = args.n_samples
imshow_size = args.imshow_size
latent_dim =  args.latent_dim
truncation =  args.truncation
n_test=  args.n_test 
id_dir = args.id_dir
sample_z_path = args.sample_z_path
save_iterations_path = args.save_iterations_path
mask_guided_iterations = args.mask_guided_iterations 
lr = args.lr
n_mean_latent = args.n_mean_latent

###############################################
save_path = save_path_root+str(id_dir)+"/"

if sample_z_path:
    save_path = save_path_root+sample_z_path.split('/')[-1].split('.')[0]+"/"

if not os.path.exists(save_path):
    os.makedirs(save_path)
#############################################

if save_iterations_path:
    iterations_path = save_path+'iterations/'

    if not os.path.exists(iterations_path):
        os.makedirs(iterations_path)
#############################################


device = 'cuda:0'

generator = Generator(image_size, latent_dim, 8)
generator_ckpt = torch.load(generator_path, map_location='cpu')
generator.load_state_dict(generator_ckpt["g_ema"], strict=False)
generator.eval().to(device)
print(f'[StyleGAN2 generator loaded] {generator_path}\n')

classes = ['background','semantic_part']

with torch.no_grad():
    trunc_mean = generator.mean_latent(4096).detach().clone()
    latent = generator.get_latent(torch.randn(n_samples, latent_dim, device=device))
    imgs_gen, features = generator([latent],
                                   truncation=truncation,
                                   truncation_latent=trunc_mean,
                                   input_is_latent=True,
                                   randomize_noise=True)
    torch.cuda.empty_cache()

@torch.no_grad()
def concat_features(features):
    h = max([f.shape[-2] for f in features])
    w = max([f.shape[-1] for f in features])
    return torch.cat([torch.nn.functional.interpolate(f, (h,w), mode='nearest') for f in features], dim=1)

data = dict(
    features=concat_features(features).cpu(),
)

print(' data[features].shape[1] is: {}'.format(data['features'].shape[1]))  #5376
print('Start Testing')
device = 'cuda' 
generator.eval().to(device) 
print('loading the model')
net = FewShotCNN(data['features'].shape[1], len(classes), size='S')
net.load_state_dict(torch.load(save_segmentation_model_path+segmentation_part+'.pt'))
net.eval().to('cpu')
print('model is loaded')

with torch.no_grad():
   
    sample_z = torch.randn(n_test, latent_dim, device=device)
    
    if sample_z_path:
        sample_z = torch.load(sample_z_path)

    torch.save(sample_z,save_path+"sample_z.pt")
    
    imgs_gen, features = generator([sample_z.to(device)],   
                                   truncation=truncation,
                                   truncation_latent=trunc_mean.to(device),  
                                   input_is_latent=False,
                                   randomize_noise=False)
    torch.cuda.empty_cache()


    source_features_tens = [torch.tensor(f, device ='cpu') for f in features]
    out = net(concat_features(source_features_tens))
    predictions = out.data.max(1)[1].cpu().numpy()

    masks = np.zeros((n_test,image_size, image_size, len(classes)))

    for i in range(n_test):
        for c in range(len(classes)):
            masks[i,:,:,c] = (predictions[i,:,:]==c)

    targ_generator = Generator(image_size, latent_dim, 8).to(device)
    targ_generator = nn.parallel.DataParallel(targ_generator)

    targ_generator_ckpt = torch.load(target_model_path)
    targ_generator.load_state_dict(targ_generator_ckpt["g_ema"], strict=False)
    targ_generator.eval().to(device)
    print(f'[StyleGAN2 generator for target style loaded] {target_model_path}\n')

    ########## target generator setup
    targ_truncation =  float(1)
    targ_mean_latent = None

    targ_imgs_gen, targ_features = targ_generator([sample_z.to(device)],
                                   truncation=targ_truncation,
                                   truncation_latent=targ_mean_latent, 
                                   input_is_latent=False,
                                   randomize_noise=False)

    torch.cuda.empty_cache()

    targ_features_tens = [torch.tensor(t, device ='cpu') for t in targ_features]
    target_out = net(concat_features(targ_features_tens))
    targ_predictions = target_out.data.max(1)[1].cpu().numpy()

    targ_masks = np.zeros((n_test,image_size, image_size, len(classes)))

    for i in range(n_test):
        for c in range(len(classes)):
            targ_masks[i,:,:,c] = (targ_predictions[i,:,:]==c)


    #####################################################

    img_name = save_path+"org_source.png"
    img_tens = (imgs_gen.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy())*0.5 + 0.5
    pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
    pil_img.save(img_name)

    img_name = save_path+"org_target.png"
    img_tens = (targ_imgs_gen.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy())*0.5 + 0.5
    pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
    pil_img.save(img_name)

#################################################################################################

def caluclate_loss(synth_img,img,perceptual_net,mask,MSE_Loss,image_resolution): 
   
    img_p = torch.nn.Upsample(scale_factor=(256/image_resolution), mode='bilinear')(img)
    real_0,real_1,real_2,real_3=perceptual_net(img_p)
    synth_p= torch.nn.Upsample(scale_factor=(256/image_resolution), mode='bilinear')(synth_img) 
    synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_p)

    perceptual_loss=0
    mask= torch.nn.Upsample(scale_factor=(256/image_resolution), mode='bilinear')(mask)    
    perceptual_loss+=MSE_Loss(synth_0*mask.expand(1,64,256,256),real_0*mask.expand(1,64,256,256))
    perceptual_loss+=MSE_Loss(synth_1*mask.expand(1,64,256,256),real_1*mask.expand(1,64,256,256))
    mask= torch.nn.Upsample(scale_factor=(64/256), mode='bilinear')(mask)
    perceptual_loss+=MSE_Loss(synth_2*mask.expand(1,256,64,64),real_2*mask.expand(1,256,64,64))
    mask= torch.nn.Upsample(scale_factor=(32/64), mode='bilinear')(mask) 
    perceptual_loss+=MSE_Loss(synth_3*mask.expand(1,512,32,32),real_3*mask.expand(1,512,32,32))
    
    return perceptual_loss

#################################################################################################
def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)
#################################################################################################

torch.cuda.empty_cache()
image_resolution = image_size 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#################################################################################################

transform = transforms.Compose([transforms.ToTensor()])
img_source = imgs_gen[0,:,:,:].unsqueeze(0).to(device)    #(1,3,image_size,image_size) (1,3,256,256)
img_target = targ_imgs_gen[0,:,:,:].unsqueeze(0).to(device)   #(1,3,image_size,image_size) (1,3,256,256)
mask = transforms.ToTensor()(masks[0,:,:,-1]).unsqueeze(0).to(device)  #(1,3,image_size,image_size) (1,3,256,256)
targ_mask = transforms.ToTensor()(targ_masks[0,:,:,-1]).unsqueeze(0).to(device)  #(1,3,image_size,image_size) (1,3,256,256)

mask_0 = mask[:,0,:,:].unsqueeze(0) #(1,1,image_resolution,image_resolution)
mask_1 = mask_0.clone()
mask_1 = 1-(mask_1)  #(1,1,image_resolution,image_resolution)

targ_mask_0 = targ_mask[:,0,:,:].unsqueeze(0) #(1,1,image_resolution,image_resolution)
targ_mask_1 = targ_mask_0.clone()
targ_mask_1 = 1-(targ_mask_1) #(1,1,image_resolution,image_resolution)

#################################################################################################

cross_over_source = (img_source*mask_1)+(img_target*targ_mask_0)
cross_over_source_image = tensor2image(cross_over_source.squeeze(0).to('cpu'))

cross_over_target = (img_source*mask_0)+(img_target*targ_mask_1)
cross_over_target_image = tensor2image(cross_over_target.squeeze(0).to('cpu'))

img_name = save_path+"naive_crossover_source.png"
img_tens = (cross_over_source.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy())*0.5 + 0.5
pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
pil_img.save(img_name)

img_name = save_path+"naive_crossover_target.png"
img_tens = (cross_over_target.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy())*0.5 + 0.5
pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
pil_img.save(img_name)

img_name = save_path+"mask_source_0.png"
img_tens = mask_0[0,0,:,:].cpu().numpy()
pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
pil_img.save(img_name)

img_name = save_path+"mask_source_1.png"
img_tens = mask_1[0,0,:,:].cpu().numpy()
pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
pil_img.save(img_name)

img_name = save_path+"mask_target_0.png"
img_tens = targ_mask_0[0,0,:,:].cpu().numpy()
pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
pil_img.save(img_name)

img_name = save_path+"mask_target_1.png"
img_tens = targ_mask_1[0,0,:,:].cpu().numpy()
pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
pil_img.save(img_name)

print("naive_crossover_source_target images are saved")

###############################################################################

g_ema = generator
with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)  

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

print("latent works")

noises_single = g_ema.make_noise() 
noises = []
for noise in noises_single:
    noises.append(noise.repeat(img_source.shape[0], 1, 1, 1).normal_())

latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(img_source.shape[0], 1)
latent_in_1 = latent_in

latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

latent_in.requires_grad = True

for noise in noises:
    noise.requires_grad = True

perceptual_net=VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device) #conv1_1,conv1_2,conv2_2,conv3_3

MSE_Loss=nn.MSELoss(reduction="mean")
optimizer = optim.Adam([latent_in],lr=lr)

print("Start embeding mask on target and source images")
loss_list=[]
latent_path = []

for i in range(mask_guided_iterations):
    t = i / mask_guided_iterations
    optimizer.param_groups[0]["lr"] = lr

    synth_img, _ = g_ema([latent_in], input_is_latent=True, noise=noises)

    batch, channel, height, width = synth_img.shape

    if height > image_size:
        factor = height // image_size

        synth_img = synth_img.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        synth_img = synth_img.mean([3, 5])

    loss_wl1=caluclate_loss(synth_img,img_source,perceptual_net,mask_1,MSE_Loss,image_size)
    loss_wl0=caluclate_loss(synth_img,img_target,perceptual_net,mask_0,MSE_Loss,image_size)
    mse_w0 = F.mse_loss(synth_img*mask_1.expand(1,3,image_size,image_size),img_source*mask_1.expand(1,3,image_size,image_size))
    mse_w1 = F.mse_loss(synth_img*mask_0.expand(1,3,image_size,image_size),img_target*mask_0.expand(1,3,image_size,image_size)) 
    mse_crossover = 3*(F.mse_loss(synth_img.float(),cross_over_source.float()))
    p_loss=2*((loss_wl0)+loss_wl1)
    mse_loss = (mse_w0)+mse_w1
    loss = (p_loss)+(mse_loss)+(mse_crossover)

    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
    
    noise_normalize_(noises)

    lr_schedule = optimizer.param_groups[0]['lr']


    if (i + 1) % 100 == 0:
        latent_path.append(latent_in.detach().clone())

        loss_np=loss.detach().cpu().numpy()
        loss_0=loss_wl0.detach().cpu().numpy()
        loss_1=loss_wl1.detach().cpu().numpy()
        mse_0=mse_w0.detach().cpu().numpy()
        mse_1=mse_w1.detach().cpu().numpy()
        mse_loss = mse_loss.detach().cpu().numpy()
    
        print("iter{}: loss -- {},  loss0 --{},  loss1 --{}, mse0--{}, mse1--{}, mseTot--{}, lr--{}".format(i,loss_np,loss_0,loss_1,mse_0,mse_1,mse_loss,lr_schedule))

        if save_iterations_path:
            img_name = iterations_path+"{}_D1.png".format(str(i).zfill(6))    
            img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises) 
            img_tens = (img_gen.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy())*0.5 + 0.5
            pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
            pil_img.save(img_name)

    if i == (mask_guided_iterations-1):
        img_name = save_path+"{}_D1.png".format(str(i).zfill(6))
        img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)
        img_tens = (img_gen.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy())*0.5 + 0.5
        pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
        pil_img.save(img_name)

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

g_ema = generator
with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)  
        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

print("latent works")

noises_single = g_ema.make_noise() 
noises = []
for noise in noises_single:
    noises.append(noise.repeat(img_source.shape[0], 1, 1, 1).normal_())

latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(img_source.shape[0], 1)
latent_in_1 = latent_in
latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

latent_in.requires_grad = True

for noise in noises:
    noise.requires_grad = True

########################################################################################################
perceptual_net=VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device) 
MSE_Loss=nn.MSELoss(reduction="mean")
optimizer = optim.Adam([latent_in],lr=lr)

print("Start embeding mask on target and source images")
loss_list=[]
latent_path = []

for i in range(mask_guided_iterations):
    t = i / mask_guided_iterations
    optimizer.param_groups[0]["lr"] = lr
    synth_img, _ = g_ema([latent_in], input_is_latent=True, noise=noises)
    batch, channel, height, width = synth_img.shape

    if height > image_size:
        factor = height // image_size

        synth_img = synth_img.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        synth_img = synth_img.mean([3, 5])
              
    loss_wl1=caluclate_loss(synth_img,img_source,perceptual_net,mask_0,MSE_Loss,image_size)
    loss_wl0=caluclate_loss(synth_img,img_target,perceptual_net,targ_mask_1,MSE_Loss,image_size)
    mse_w0 = F.mse_loss(synth_img*mask_0.expand(1,3,image_size,image_size),img_source*mask_0.expand(1,3,image_size,image_size))
    mse_w1 = F.mse_loss(synth_img*targ_mask_1.expand(1,3,image_size,image_size),img_target*targ_mask_1.expand(1,3,image_size,image_size))    
    mse_crossover = 3*(F.mse_loss(synth_img.float(),cross_over_target.float()))
    p_loss=2*(loss_wl0+(loss_wl1))
    mse_loss = mse_w0+(mse_w1)
    loss = (p_loss)+(mse_loss)+(mse_crossover)
    optimizer.zero_grad()
    loss.backward()  
    optimizer.step()
    

    noise_normalize_(noises)

    lr_schedule = optimizer.param_groups[0]['lr']


    if (i + 1) % 100 == 0:
        latent_path.append(latent_in.detach().clone())
        loss_np=loss.detach().cpu().numpy()
        loss_0=loss_wl0.detach().cpu().numpy()
        loss_1=loss_wl1.detach().cpu().numpy()
        mse_0=mse_w0.detach().cpu().numpy()
        mse_1=mse_w1.detach().cpu().numpy()
        mse_loss = mse_loss.detach().cpu().numpy()
    
        print("iter{}: loss -- {},  loss0 --{},  loss1 --{}, mse0--{}, mse1--{}, mseTot--{}, lr--{}".format(i,loss_np,loss_0,loss_1,mse_0,mse_1,mse_loss,lr_schedule))
 
        if save_iterations_path:
            img_name = iterations_path+"{}_D2.png".format(str(i).zfill(6))    
            img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)  
            img_tens = (img_gen.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy())*0.5 + 0.5
            pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
            pil_img.save(img_name)

    if i == (mask_guided_iterations-1):
        img_name = save_path+"{}_D2.png".format(str(i).zfill(6))
        img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)
        img_tens = (img_gen.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy())*0.5 + 0.5
        pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
        pil_img.save(img_name)
