# This code is released under the CC BY-SA 4.0 license.

import torch
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
from torchvision.transforms import v2
from kornia.filters import SpatialGradient, SpatialGradient3d, filter3d, filter2d
import torchvision
import nibabel as nib
from pathlib import Path
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


def NGF_loss(model, fake_B, real_A):
    # as the training allows 2D and 3D training
    if model.dim_ == 2:
        SpatialGradient_ = SpatialGradient
    else:
        SpatialGradient_ = SpatialGradient3d

    grad_src = SpatialGradient_()(fake_B)  # getting gradient of the source image
    grad_tgt = SpatialGradient_()(real_A)  # getting gradient of the target image

    gradmag_src = torch.sqrt(torch.sum(torch.pow(grad_src, 2), dim=2) + model.epsilonT + 1e-5)  # calculating norm of the gradient of the source image
    gradmag_tgt = torch.sqrt(torch.sum(torch.pow(grad_tgt, 2), dim=2) + model.epsilonT + 1e-5)  # calculating norm of the gradient of the target image

    # calculating pixel wise NGF loss
    NGF = torch.sum(grad_src * grad_tgt, dim=2) / ((gradmag_src) * (gradmag_tgt))
    NGF_L = 1 - 1 / 2 * torch.pow(NGF, 2)

    # calculating mean
    NGFM = torch.mean(NGF_L)
    return NGFM


class NGFModel(BaseModel):
    """
    This class implements the NGF model, for learning image-to-image translation without paired data.
    This implementation uses the CycleGAN implementation for fair comparisons.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--epsilonT', type=float, default=0.0001, help='tolerance parameter for NGF loss')
        parser.add_argument('--multi_scale_discriminator', action='store_true', default=False)
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_NGF', type=float, default=10.0, help='weight for the edge loss')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for an identity multiplied by the `lambda_NGF` loss B = gen(B)')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.multi_scale_discriminator = opt.multi_scale_discriminator
        self.epsilonT = opt.epsilonT

        '''
        loss values names
        D_A : discriminator image domain
        G_A : GAN image domain
        NGF_A : NGF loss image domain
        idt_A : identity loss image domain
        '''
        self.loss_names = ['D_A', 'G_A', 'NGF_A', 'idt_A']

        # networks names, during train this will be saved
        if self.isTrain:
            self.model_names = ['G_A', 'D_A', 'optimizer_D', 'optimizer_G', 'schedulers']
        else:
            # during test time, only load Gs
            self.model_names = ['G_A']

        '''
        G_A: Generator
        D_A: image domain discriminator
        D_B: bone discriminator (optional)
        '''
        # defining generator
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, dim=opt.dim, opt=opt)

        if self.isTrain:  # discriminators used only during training
            # defining image domain discriminator
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, dim=opt.dim)
            self.netD_A.cuda()
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
            SRC = Source
            TGT = Target
            dim_ = model dimension, this training allows 3D training
        """
        if self.dim_ == 2:
            self.real_A = input['SRC'][:, :, :, :, 0].to(self.device).float()
            self.real_B = input['TGT'][:, :, :, :, 0].to(self.device).float()
        else:
            self.real_A = input['SRC'].permute((0, 1, 4, 2, 3)).to(self.device).float()
            self.real_B = input['TGT'].permute((0, 1, 4, 2, 3)).to(self.device).float()

    def forward(self):
        """Run forward pass; called by function <optimize_parameters>"""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for a single discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        if self.multi_scale_discriminator:
            loss_D = netD.calc_dis_loss(fake.detach(), real)
        else:
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(pred_real, True)
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        """Calculate GAN loss for all discriminators"""
        # from CycleGAN implementation
        fake_B = self.fake_B_pool.query(self.fake_B)
        # calculating image domain discriminator loss
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_G(self):
        """Calculate the loss for generator G_A"""
        # getting lambdas
        lambda_idt = self.opt.lambda_identity
        lambda_NGF = self.opt.lambda_NGF

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_NGF * lambda_idt
        else:
            self.loss_idt_A = 0

        # GAN loss D_A(G_A(A))
        if self.multi_scale_discriminator:
            self.loss_G_A = self.netD_A.calc_gen_loss(self.fake_B)
        else:
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # NGF loss
        self.loss_NGF_A = NGF_loss(self, self.fake_B, self.real_A) * lambda_NGF

        self.loss_G = self.loss_G_A + self.loss_NGF_A + self.loss_idt_A
        self.loss_G.backward()

    def optimize_parameters(self):
        self.netG_A.cuda()
        self.netD_A.cuda()
        self.forward()
        self.set_requires_grad([self.netD_A], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A's gradients to zero
        self.backward_G()             # calculate gradients for G_A
        self.optimizer_G.step()       # update G_A's weights
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()  # set D_A's gradients to zero
        self.backward_D()             # calculate gradients for D_A
        self.optimizer_D.step()       # update D_A's weights