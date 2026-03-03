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
def get_3d_locations(b, d,h,w,device_):
      locations_x = torch.linspace(0, w-1, w).view(1, 1, 1, w).to(device_).expand(b, d, h, w) - w//2
      locations_y = torch.linspace(0, h-1, h).view(1, 1, h, 1).to(device_).expand(b, d, h, w) - h//2
      locations_z = torch.linspace(0, d-1,d).view(1, d, 1, 1).to(device_).expand(b, d, h, w) - d//2
      # stack locations
      locations_3d = torch.stack([locations_z, locations_y, locations_x], dim=4).view(-1, 3, 1)
      return locations_3d
def get_2d_locations(b, h,w,device_):
      locations_x = torch.linspace(0, w-1, w).view(1, 1, w).to(device_).expand(b, h, w) - w//2
      locations_y = torch.linspace(0, h-1, h).view(1, h, 1).to(device_).expand(b, h, w) - h//2
      # stack locations
      locations_3d = torch.stack([locations_y, locations_x], dim=3).view(-1, 2, 1)
      return locations_3d
def rotate(input_tensor, rotation_matrix):
    device_ = input_tensor.device
    b, _, d, h, w  = input_tensor.shape
    # get x,y,z indices of target 3d data
    locations_3d = get_3d_locations(b, d, h, w, device_)
    # rotate target positions to the source coordinate
    rotated_3d_positions = torch.bmm(rotation_matrix.view(1, 3, 3).expand(b*d*h*w, 3, 3).float(), locations_3d.float()).view(b, d,h,w,3)
    rot_locs = list(torch.split(rotated_3d_positions, split_size_or_sections=1, dim=4))
    rot_locs[0] = rot_locs[0]+d//2
    rot_locs[1] = rot_locs[1]+h//2
    rot_locs[2] = rot_locs[2]+w//2
    # change the range of x,y,z locations to [-1,1]
    normalised_locs_x = (2.0*rot_locs[0] - (d-1))/(d-1)
    normalised_locs_y = (2.0*rot_locs[1] - (h-1))/(h-1)
    normalised_locs_z = (2.0*rot_locs[2] - (w-1))/(w-1)
    grid = torch.stack([normalised_locs_x, normalised_locs_y, normalised_locs_z], dim=4).view(b, d, h, w, 3)
    rotated_signal = F.grid_sample(input=input_tensor+2, grid=grid, mode='bilinear',  align_corners=True)
    rotated_signal[rotated_signal <= 1] = 1
    rotated_signal = rotated_signal - 2
    return rotated_signal

    
def edge_loss(model, fake_B, real_A):
    if model.dim_ == 2:
        SpatialGradient_ = SpatialGradient
    else:
        SpatialGradient_ = SpatialGradient3d
    eps = 1e-8
    grad_src = SpatialGradient_()(fake_B)
    grad_tgt = SpatialGradient_()(real_A)
    gradmag_src = torch.sqrt(torch.sum(torch.pow(grad_src,2), dim=2)+model.alpha_NGF**2)
    gradmag_tgt = torch.sqrt(torch.sum(torch.pow(grad_tgt,2), dim=2)+model.alpha_NGF**2)
    NGF_L = 1-1/2*torch.pow(torch.sum(grad_src*grad_tgt, dim=2) / ((gradmag_src + eps) * (gradmag_tgt + eps)), 2)
    NGF_L = 1-1/2*torch.pow(torch.sum(grad_src*grad_tgt, dim=2) / ((gradmag_src + eps) * (gradmag_tgt + eps)), 2)
    NGFM = torch.mean(NGF_L)
    return NGFM

class PIX2PIXModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.add_argument('--alpha_NGF', type=float, default=0.0001, help='weight for cycle loss (A -> B -> A)')
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        
        parser.add_argument('--depth', type=int, default=4, help='number of down')
        parser.add_argument('--heads', type=int, default=12, help='number of down')
        parser.add_argument('--img_size', type=int, default=256, help='number of down')
        parser.add_argument('--fth', type=float, default=0.5, help='Foreground threshold')
        parser.add_argument('--feat_size', type=int, default=16, help='kernel size for output convolution layer')
        parser.add_argument('--out_kernel', type=int, default=7, help='kernel size for output convolution layer')
        parser.add_argument('--upsample', type=str, default='deconv', help='UNet upsampling mode')
        parser.add_argument('--vit_emb_size', type=int, default=768, help='kernel size for output convolution layer')
        parser.add_argument('--vit_norm', type=str, default='layer', help='ViT norm')
        parser.add_argument('--vit_img_size', type=int, nargs='+', default=[256, 256], help='ViT image size')
        parser.add_argument('--window_size', type=int, default=2, help='number of down')
        parser.add_argument('--patch_size', type=int, default=16, help='kernel size for output convolution layer')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.depth = opt.depth
        self.heads = opt.heads
        self.img_size = opt.img_size
        self.feat_size = opt.feat_size
        self.out_kernel = opt.out_kernel
        self.upsample = opt.upsample
        self.vit_emb_size = opt.vit_emb_size
        self.vit_norm = opt.vit_norm
        self.fth = opt.fth
        self.vit_img_size = opt.vit_img_size
        self.window_size = opt.window_size
        self.patch_size = opt.patch_size

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.alpha_NGF = opt.alpha_NGF
        self.loss_names = ['D_A', 'G_A', 'L1_A', 'idt_A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')

        self.visual_names = visual_names_A # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'D_A', 'optimizer_D', 'optimizer_G', 'schedulers']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, dim = opt.dim, opt=opt)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, dim = opt.dim)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.input_ = input
        AtoB = self.opt.direction == 'AtoB'
        if self.dim_ == 2: 
            self.real_A = input['SRC'][:,:,:,:,0].to(self.device).float()
            self.real_B = input['TGT'][:,:,:,:,0].to(self.device).float()
        else:
            self.real_A = input['SRC'].permute((0,1,4,2,3)).to(self.device).float()
            self.real_B = input['TGT'].permute((0,1,4,2,3)).to(self.device).float()
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_L1_A = self.criterionIdt(self.fake_B, self.real_B) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_G = self.loss_G_A+  self.loss_L1_A + self.loss_idt_A
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.netG_A.cuda()
        self.netD_A.cuda()
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
