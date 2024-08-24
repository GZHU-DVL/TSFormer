import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from tqdm import tqdm
from imageio import imsave

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision.utils import save_image

from models.generator.TSFormer_Full import TSFormer_Full
from datasets.dataset import create_image_dataset
from options.test_options import TestOptions
from utils.misc import sample_data, postprocess
from torchvision.transforms.functional import crop

is_cuda = torch.cuda.is_available()
if is_cuda:
    print('Cuda is available')
    cudnn.enable = True
    cudnn.benchmark = True

opts = TestOptions().parse

os.makedirs('{:s}'.format(opts.result_root), exist_ok=True)

# model & load model
generator = TSFormer_Full(inp_channels=3, out_channels=3)
if opts.pre_trained != '':
    generator.load_state_dict(torch.load(opts.pre_trained)['generator'])
else:
    print('Please provide pre-trained model!')

if is_cuda:
    generator = generator.cuda()

# dataset
image_dataset = create_image_dataset(opts)
image_data_loader = data.DataLoader(
    image_dataset,
    batch_size=opts.batch_size,
    shuffle=False,
    num_workers=opts.num_workers,
    drop_last=False
)
image_data_loader = sample_data(image_data_loader)

def overlapping_grid_indices(x_cond, output_size, r=None):# 得到随机裁剪的开始位置
    _, c, h, w = x_cond.shape
    r = 16 if r is None else r
    h_list = [i for i in range(0, h - output_size + 1, r)]
    w_list = [i for i in range(0, w - output_size + 1, r)]
    return h_list, w_list

print('start test...')
p_size = 256
with torch.no_grad():

    generator.eval()
    for _ in tqdm(range(opts.number_eval)):

        ground_truth, mask, edge, gray_image, hog_image = next(image_data_loader)
        if is_cuda:
            ground_truth, mask, edge, gray_image, hog_image = ground_truth.cuda(), mask.cuda(), edge.cuda(), gray_image.cuda(), hog_image.cuda()

        input_image, input_edge, input_gray_image, input_hog = ground_truth * mask, edge * mask, gray_image * mask, hog_image * mask

        h_list, w_list = overlapping_grid_indices(input_image, output_size=256, r=16)

        corners = [(i, j) for i in h_list for j in w_list] #得到开始裁剪的位置坐标

        data_grid_mask = torch.zeros_like(input_image, device=input_image.device)
        for (hi, wi) in corners:
            data_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        manual_batching_size = 16
        data_input_cond_patch = torch.cat([crop(input_image, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)# 将随机裁剪得到的图像 组成一个batch
        data_edge_cond_patch = torch.cat([crop(input_edge, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)# 将随机裁剪得到的图像 组成一个batch
        data_gray_cond_patch = torch.cat([crop(input_gray_image, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)# 将随机裁剪得到的图像 组成一个batch
        data_hog_cond_patch = torch.cat([crop(input_hog, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)# 将随机裁剪得到的图像 组成一个batch
        data_output = torch.zeros_like(input_image, device=input_image.device)

        for i in range(0, len(corners), manual_batching_size):
            output, projected_edge, projected_edge_first, projected_hog = generator(data_input_cond_patch[i:i + manual_batching_size],
                                                                            torch.cat(
                                                                                (data_edge_cond_patch[i:i + manual_batching_size], data_gray_cond_patch[i:i + manual_batching_size]),
                                                                                dim=1),
                                                                            data_hog_cond_patch[i:i + manual_batching_size])
            # outputs = model(data_input_cond_patch[i:i + manual_batching_size])
            for idx, (hi, wi) in enumerate(corners[i:i + manual_batching_size]):
                data_output[0, :, hi:hi + p_size, wi:wi + p_size] += output[idx]

        result = torch.div(data_output, data_grid_mask)

        output_comp = ground_truth * mask + result * (1 - mask)
        
        output_comp = postprocess(output_comp)

        output_comp_gt = postprocess(ground_truth)

        # ground_truth = postprocess(ground_truth)

        output_comp_masked = output_comp * mask
    
        # save_image(output_comp, opts.result_root + '/{:03d}_im.png'.format(_+1))  # For Paris StreetView

        # save_image(output_comp, opts.result_root + '/{:06d}.png'.format(_ + 182637))  # For CelebA

        save_image(output_comp, opts.result_root + '/{:05d}.png'.format(_))  # For Places2
        save_image(output_comp_masked, opts.result_root + '/{:05d}m.png'.format(_))
        save_image(output_comp_gt, opts.result_root + '/{:05d}g.png'.format(_))
