import os

import torch
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils.distributed import get_rank, reduce_loss_dict
from utils.misc import requires_grad, sample_data
from criteria.loss_full import generator_loss_func, discriminator_loss_func


def train(opts, image_data_loader, generator, discriminator, extractor, generator_optim, discriminator_optim, is_cuda):
    image_data_loader = sample_data(image_data_loader)
    pbar = range(opts.train_iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opts.start_iter, dynamic_ncols=True, smoothing=0.01)

    if opts.distributed:
        generator_module, discriminator_module = generator.module, discriminator.module
    else:
        generator_module, discriminator_module = generator, discriminator

    writer = SummaryWriter(opts.log_dir)

    for index in pbar:

        i = index + opts.start_iter
        if i > opts.train_iter:
            print('Done...')
            break

        ground_truth, mask, edge, gray_image, hog_image = next(image_data_loader)

        if is_cuda:
            ground_truth, mask, edge, gray_image, hog_image = ground_truth.cuda(), mask.cuda(), \
                                                              edge.cuda(), gray_image.cuda(), hog_image.cuda()

        input_image, input_edge, input_gray_image, input_hog = ground_truth * mask, edge * mask, gray_image * mask, hog_image * mask

        # ---------
        # Generator
        # ---------
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        output, projected_edge, projected_edge_first, projected_hog = generator(input_image,
                                                                 torch.cat((input_edge, input_gray_image), dim=1),
                                                                 input_hog)

        comp = ground_truth * mask + output * (1 - mask)

        fake_output = discriminator(output)

        vgg_comp, vgg_output, vgg_ground_truth = extractor(comp), extractor(output), extractor(ground_truth)

        generator_loss_dict = generator_loss_func(
            mask=mask, output=output, ground_truth=ground_truth, edge=edge,
            vgg_comp=vgg_comp, vgg_output=vgg_output, vgg_ground_truth=vgg_ground_truth, fake_output=fake_output,
            projected_edge=projected_edge, projected_edge_first=projected_edge_first,
            hog_image=hog_image, projected_hog=projected_hog
        )
        generator_loss = generator_loss_dict['loss_hole'] * opts.HOLE_LOSS + \
                         generator_loss_dict['loss_valid'] * opts.VALID_LOSS + \
                         generator_loss_dict['loss_perceptual'] * opts.PERCEPTUAL_LOSS + \
                         generator_loss_dict['loss_style'] * opts.STYLE_LOSS + \
                         generator_loss_dict['loss_adversarial'] * opts.ADVERSARIAL_LOSS + \
                         generator_loss_dict['loss_intermediate'] * opts.INTERMEDIATE_LOSS
        generator_loss_dict['loss_joint'] = generator_loss

        generator_optim.zero_grad()
        generator_loss.backward()
        generator_optim.step()

        # -------------
        # Discriminator
        # -------------
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        D_real = discriminator(ground_truth)
        D_fake = discriminator(output.detach())

        discriminator_loss_dict = discriminator_loss_func(D_real, D_fake)
        discriminator_loss = discriminator_loss_dict['loss_adversarial']
        discriminator_loss_dict['loss_joint'] = discriminator_loss

        discriminator_optim.zero_grad()
        discriminator_loss.backward()
        discriminator_optim.step()

        # ---
        # log
        # ---
        generator_loss_dict_reduced, discriminator_loss_dict_reduced = reduce_loss_dict(
            generator_loss_dict), reduce_loss_dict(discriminator_loss_dict)

        pbar_g_loss_hole = generator_loss_dict_reduced['loss_hole'].mean().item()
        pbar_g_loss_valid = generator_loss_dict_reduced['loss_valid'].mean().item()
        pbar_g_loss_perceptual = generator_loss_dict_reduced['loss_perceptual'].mean().item()
        pbar_g_loss_style = generator_loss_dict_reduced['loss_style'].mean().item()
        pbar_g_loss_adversarial = generator_loss_dict_reduced['loss_adversarial'].mean().item()
        pbar_g_loss_intermediate = generator_loss_dict_reduced['loss_intermediate'].mean().item()
        pbar_g_loss_joint = generator_loss_dict_reduced['loss_joint'].mean().item()

        pbar_d_loss_adversarial = discriminator_loss_dict_reduced['loss_adversarial'].mean().item()
        pbar_d_loss_joint = discriminator_loss_dict_reduced['loss_joint'].mean().item()

        if get_rank() == 0:

            pbar.set_description((
                f'g_loss_joint: {pbar_g_loss_joint:.4f} '
                f'd_loss_joint: {pbar_d_loss_joint:.4f}'
            ))

            writer.add_scalar('g_loss_hole', pbar_g_loss_hole, i)
            writer.add_scalar('g_loss_valid', pbar_g_loss_valid, i)
            writer.add_scalar('g_loss_perceptual', pbar_g_loss_perceptual, i)
            writer.add_scalar('g_loss_style', pbar_g_loss_style, i)
            writer.add_scalar('g_loss_adversarial', pbar_g_loss_adversarial, i)
            writer.add_scalar('g_loss_intermediate', pbar_g_loss_intermediate, i)
            writer.add_scalar('g_loss_joint', pbar_g_loss_joint, i)

            writer.add_scalar('d_loss_adversarial', pbar_d_loss_adversarial, i)
            writer.add_scalar('d_loss_joint', pbar_d_loss_joint, i)

            if i % opts.save_interval == 0:
                torch.save(
                    {
                        'n_iter': i,
                        'generator': generator_module.state_dict(),
                        'discriminator': discriminator_module.state_dict(),
                        'generator_optim': generator_optim.state_dict(),
                        'discriminator_optim': discriminator_optim.state_dict()
                    },
                    f"{opts.save_dir}/{str(i).zfill(6)}.pt",
                )
