import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import gram_matrix


####################################################################################################
# adversarial loss for different gan mode
####################################################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real examples or fake examples

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def calculate_loss(self, prediction, target_is_real, is_dis=False):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real examples or fake examples

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
            if self.gan_mode == 'lsgan':
                loss = loss * 0.5
        else:
            if is_dis:
                if target_is_real:
                    prediction = -prediction
                if self.gan_mode == 'wgangp':
                    loss = prediction.mean()
                elif self.gan_mode == 'nonsaturating':
                    loss = F.softplus(prediction).mean()
                elif self.gan_mode == 'hinge':
                    loss = self.loss(1 + prediction).mean()
            else:
                if self.gan_mode == 'nonsaturating':
                    loss = F.softplus(-prediction).mean()
                else:
                    loss = -prediction.mean()
        return loss

    def __call__(self, predictions, target_is_real, is_dis=False):
        """Calculate loss for multi-scales gan"""
        if isinstance(predictions, list):
            losses = []
            for prediction in predictions:
                losses.append(self.calculate_loss(prediction, target_is_real, is_dis))
            loss = sum(losses)
        else:
            loss = self.calculate_loss(predictions, target_is_real, is_dis)

        return loss


def generator_loss_func(
        mask, output, ground_truth, edge,
        vgg_comp, vgg_output, vgg_ground_truth, fake_output, projected_edge, projected_edge_first,
        hog_image, projected_hog):
    l1 = nn.L1Loss()
    criterion = nn.BCELoss()

    # ---------
    # hole loss
    # ---------
    loss_hole = l1((1 - mask) * output, (1 - mask) * ground_truth)

    # ----------
    # valid loss
    # ----------
    loss_valid = l1(mask * output, mask * ground_truth)

    # ---------------
    # perceptual loss
    # ---------------
    loss_perceptual = 0.0
    for i in range(3):
        loss_perceptual += l1(vgg_output[i], vgg_ground_truth[i])
        loss_perceptual += l1(vgg_comp[i], vgg_ground_truth[i])

    # ----------
    # style loss
    # ----------
    loss_style = 0.0
    for i in range(3):
        loss_style += l1(gram_matrix(vgg_output[i]), gram_matrix(vgg_ground_truth[i]))
        loss_style += l1(gram_matrix(vgg_comp[i]), gram_matrix(vgg_ground_truth[i]))

    # ----------------
    # adversarial loss
    # ----------------
    if torch.cuda.is_available():
        GANloss = GANLoss('nonsaturating').cuda()
    loss_G_GAN = GANloss(fake_output, True) * 1.0

    # -----------------
    # intermediate loss
    # -----------------
    loss_intermediate = 0.0
    loss_intermediate += criterion(projected_edge_first, edge)
    loss_intermediate += criterion(projected_edge, edge)
    loss_intermediate += (l1(projected_hog, hog_image) * 0.1)

    return {
        'loss_hole': loss_hole.mean(),
        'loss_valid': loss_valid.mean(),
        'loss_perceptual': loss_perceptual.mean(),
        'loss_style': loss_style.mean(),
        'loss_adversarial': loss_G_GAN.mean(),
        'loss_intermediate': loss_intermediate.mean()
    }


def discriminator_loss_func(D_real, D_fake):
    if torch.cuda.is_available():
        GANloss = GANLoss('nonsaturating').cuda()
    loss_D_real = GANloss(D_real, True, is_dis=True)
    loss_D_fake = GANloss(D_fake, False, is_dis=True)
    loss_adversarial = loss_D_real + loss_D_fake

    return {
        'loss_adversarial': loss_adversarial.mean()
    }
