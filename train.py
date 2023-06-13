import warnings
warnings.filterwarnings("ignore")

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

from models.discriminator.style_discriminator import StyleDiscriminator
from models.generator.TSFormer_Full import TSFormer_Full
from models.generator.vgg16 import VGG16FeatureExtractor
from options.train_options import TrainOptions
from datasets.dataset import create_image_dataset
from utils.distributed import synchronize
from utils.ddp import data_sampler
from trainer_full import train
from models.discriminator import base_function
from utils.seedUtil import set_seed


opts = TrainOptions().parse

os.makedirs('{:s}'.format(opts.save_dir), exist_ok=True)

set_seed(42)

is_cuda = torch.cuda.is_available()
if is_cuda:
    
    print('Cuda is available')
    cudnn.enable = True
    cudnn.benchmark = True

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print('GPU number: ', n_gpu)
    opts.distributed = n_gpu > 1
    if opts.distributed:
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://localhost:57777")
        synchronize()

# model & load model
# TSFormer
generator = TSFormer_Full(inp_channels=3, out_channels=3)
# StyleDiscriminator
style_discriminator = StyleDiscriminator(256, ndf=32)
discriminator = base_function.init_net(style_discriminator, opts.init_type, opts.init_gain, initialize_weights=False)
extractor = VGG16FeatureExtractor()

# cuda
if is_cuda:
    generator, discriminator, extractor = generator.cuda(), discriminator.cuda(), extractor.cuda()

# optimizer
if opts.finetune == True:
    print('Fine tune...')
    lr = opts.lr_finetune
    generator.freeze_ec_bn = True
else:
    lr = opts.gen_lr

generator_optim = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
discriminator_optim = optim.Adam(discriminator.parameters(), lr=lr * 2, betas=(0.5, 0.9))

# load checkpoints
if opts.pre_trained != '':
    ckpt_dict = torch.load(opts.pre_trained, map_location=lambda storage, loc: storage)
    opts.start_iter = ckpt_dict['n_iter']
    generator.load_state_dict(ckpt_dict['generator'])
    discriminator.load_state_dict(ckpt_dict['discriminator'])
    # add new generator_optim and discriminator_optim
    generator_optim.load_state_dict(ckpt_dict['generator_optim'])
    discriminator_optim.load_state_dict(ckpt_dict['discriminator_optim'])

    print('Starting from iter', opts.start_iter)
else:
    print('Starting from iter', opts.start_iter)


if opts.distributed:

    generator = nn.parallel.DistributedDataParallel(
        generator, 
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        broadcast_buffers=False,
        # find_unused_parameters=True
    )
    discriminator = nn.parallel.DistributedDataParallel(
        discriminator, 
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        broadcast_buffers=False,
    )

# dataset
image_dataset = create_image_dataset(opts)
print(image_dataset.__len__())

image_data_loader = data.DataLoader(
    image_dataset,
    batch_size=opts.batch_size,
    sampler=data_sampler(
        image_dataset, shuffle=True, distributed=opts.distributed
    ),
    drop_last=True,
    num_workers=4
)

# training
train(opts, image_data_loader, generator, discriminator, extractor, generator_optim, discriminator_optim, is_cuda)



