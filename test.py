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

print('start test...')
with torch.no_grad():

    generator.eval()
    for _ in tqdm(range(opts.number_eval)):

        ground_truth, mask, edge, gray_image, hog_image = next(image_data_loader)
        if is_cuda:
            ground_truth, mask, edge, gray_image, hog_image = ground_truth.cuda(), mask.cuda(), edge.cuda(), gray_image.cuda(), hog_image.cuda()

        input_image, input_edge, input_gray_image, input_hog = ground_truth * mask, edge * mask, gray_image * mask, hog_image * mask

        output, projected_edge, projected_edge_first, projected_hog = generator(input_image,
                                                                                torch.cat(
                                                                                    (input_edge, input_gray_image),
                                                                                    dim=1),
                                                                                input_hog)
        output_comp = ground_truth * mask + output * (1 - mask)
        
        output_comp = postprocess(output_comp)
    
        save_image(output_comp, opts.result_root + '/{:03d}_im.png'.format(_+1))  # For Paris StreetView

        # save_image(output_comp, opts.result_root + '/{:06d}.png'.format(_ + 182637))  # For CelebA

        # save_image(output_comp, opts.result_root + '/{:05d}.png'.format(_))  # For Places2
