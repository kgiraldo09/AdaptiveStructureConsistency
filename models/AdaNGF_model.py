'''
Creative Commons Attribution-NonCommercial 4.0 International

Copyright (c) 2026 Kevin Estiven Giraldo Paniagua

This work is licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License.

You are free to:
Share — copy and redistribute the material in any medium or format
Adapt — remix, transform, and build upon the material

Under the following terms:
Attribution — You must give appropriate credit, provide a link to the license,
and indicate if changes were made.

NonCommercial — You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological
measures that legally restrict others from doing anything the license permits.

Full license text available at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode

'''
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



def NGF_magnitude(model, img, alpha = 0):
    if model.dim_ == 2:
        SG = SpatialGradient
    else:
        SG = SpatialGradient3d
    grad_img = SG()(img)
    grad_img_norm =  torch.sqrt(torch.sum(torch.pow(grad_img,2), dim=2))
    gradmag_img = torch.sqrt(torch.sum(torch.pow(grad_img,2), dim=2) + 1e-5 + alpha*model.multi_parameter)
    NGF = grad_img_norm / gradmag_img
    return NGF

def NGF_loss(model, fake_B, real_A, alpha = 0):
    # as the training allows 2D and 3D training
    if model.dim_ == 2:
        SpatialGradient_ = SpatialGradient
    else:
        SpatialGradient_ = SpatialGradient3d

    grad_src = SpatialGradient_()(real_A) # getting gradient of the source image
    grad_tgt = SpatialGradient_()(fake_B) # getting gradient of the target image

    gradmag_src = torch.sqrt(torch.sum(torch.pow(grad_src,2), dim=2)+1e-5 + alpha*model .multi_parameter) # calculating norm of the gradient of the source image, with the addition of the adaptive target domain tolerance parameter
    gradmag_tgt = torch.sqrt(torch.sum(torch.pow(grad_tgt,2), dim=2)+1e-5 + model.epsilonT * model.multi_parameter) # calculating norm of the gradient of the target image, with the addition of the fixed target domain tolerance parameter
    # calculating pixel wise NGF loss
    NGF = torch.sum(grad_src*grad_tgt, dim=2) / (gradmag_src * gradmag_tgt)
    NGF_L = 1-1/2*torch.pow(NGF, 2) 
    # calculating mean
    NGFM = torch.mean(NGF_L)
    return NGFM

class AdaNGFModel(BaseModel):
    """
    This class implements the AdaNGF model, for learning image-to-image translation without paired data.
    This implementation uses the CycleGAN implementation for fair comparisons
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # As the outpout of the adaptive target domain tolerance parameter generator is a sigmoid
        # a multiplicative factor is used, to increase the range of values
        parser.add_argument('--multi_parameter', type=float, default=1.000, help='multiplicative factor' \
                    'used to increase the range of values of the adaptive target domain tolerance parameter generator')
        parser.add_argument('--epsilonT', type=float, default=0.1, help='fixed target domain tolerance parameter / ')
        parser.add_argument('--GAN_edge_weight', type=float, default=1.0000, help='weight for cycle loss (A -> B -> A)')
        # activation function of adaptive target domain tolerance parameter generator, better to use sigmoid, not to touch this paramter
        parser.add_argument('--relu', action='store_true',default=False)
        # use detach toi avoid the NGF loss propagate also through Ge(s)
        parser.add_argument('--detach', action='store_false',default=True)
        # use epsilon_multi_scale to use a multi scale edge discriminator
        parser.add_argument('--epsilon_multi_scale', action='store_true',default=False)
        # use multi_scale to use a multi scale in both discriminators
        parser.add_argument('--multi_scale', action='store_true',default=False)
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_edge', type=float, default=10.0, help='weight for the edge loss')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for an identity multiplied by the lambda_edge loss B = gen(B)')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to out. The training/test scripts will call <BaseModel.get_current_losses>
        self.multi_parameter = opt.multi_parameter
        self.GAN_edge_weight = opt.GAN_edge_weight
        self.epsilonT = opt.epsilonT
        self.multi_scale = opt.multi_scale
        self.detach = opt.detach
        self.epsilon_multi_scale = opt.epsilon_multi_scale
        ''' 
        loss values names
        D_C = Discriminator edge domain
        D_A : discriminator image domain
        G_A : GAN edge domain + GAN image domain
        NGF_A : NGF loss image domain
        idt_A : identity loss image domain
        '''
        self.loss_names = ['D_C', 'D_A', 'G_A', 'NGF_A', 'idt_A']
        #  networks names, during train this will be saved
        if self.isTrain:
            self.model_names = ['G_A', 'D_A', 'D_C', 'optimizer_D', 'optimizer_D_C', 'optimizer_G', 'schedulers']
        else:  
            # during test time, only load Gs
            self.model_names = ['G_A']
        '''
        As before:
        G_A: Generator
        D_A: image domain discriminator
        D_C: edge domain discriminator
        '''
        # defining generator
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, dim = opt.dim, smish = False, opt=opt)

        if self.isTrain: # discriminator used only during training
            # defining image domain discriminator
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, dim = opt.dim, smish = False)
            self.netD_A.cuda()
            # defining edge domain discriminator
            if self.epsilon_multi_scale: # using multi-scale discriminator for the edge domain
                opt.netD = 'MsImageDisunit'
            self.netD_C = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, dim = opt.dim, smish = False)
            self.netD_C.cuda()
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_C_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            
            # define some loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_C = torch.optim.Adam(itertools.chain(self.netD_C.parameters()), lr=opt.lr if self.epsilon_multi_scale else 0.0001, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_C)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
            SRC = Source
            TGT = Target
            dim_ = model dimension, this training allows 3D training
        """
        if self.dim_ == 2: 
            self.real_A = input['SRC'][:,:,:,:,0].to(self.device).float()
            self.real_B = input['TGT'][:,:,:,:,0].to(self.device).float()
        else:
            self.real_A = input['SRC'].permute((0,1,4,2,3)).to(self.device).float()
            self.real_B = input['TGT'].permute((0,1,4,2,3)).to(self.device).float()

    def forward(self):
        """Run forward pass; called by function <optimize_parameters>"""
        # geting the G(s) and Ge(s), outA_bool allow us to decide if we ant to get Ge(s) or not
        self.fake_B, self.ada_epsitlon = self.netG_A(self.real_A, outA_bool = True)

        self.fake_NGF_A = NGF_magnitude(self, self.real_A.detach(), self.ada_epsitlon)
        self.real_NGF_B = NGF_magnitude(self, self.real_B.detach(),  self.epsilonT)


    def backward_D_basic(self, netD, real, fake, NGF=False):
        """Calculate GAN loss for a single discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        if self.multi_scale: # both multi scale
            loss_D = netD.calc_dis_loss(fake.detach(), real)
        elif self.epsilon_multi_scale and NGF: # only edge multi scale
            loss_D = netD.calc_dis_loss(fake.detach(), real)
        else: # from Cycle GAN
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
        # from cycle GAN implementation
        fake_B = self.fake_B_pool.query(self.fake_B) 
        fake_NGF_A = self.fake_C_pool.query(self.fake_NGF_A)
        # calcularting image domain loss discriminator
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        # calculating edge domain loss discriminator
        if self.GAN_edge_weight > 0:
            self.loss_D_C = self.backward_D_basic(self.netD_C, self.real_NGF_B, fake_NGF_A, NGF=True)
        else:
            self.loss_D_C = 0


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        #getting lambdas
        lambda_idt = self.opt.lambda_identity
        lambda_edge = self.opt.lambda_edge
        # Identity loss
        if lambda_idt > 0 or True:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A, self.idt_Alpha = self.netG_A(self.real_B, outA_bool = True)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_edge * lambda_idt
            idt_alpha = torch.ones_like(self.idt_Alpha) * self.epsilonT
            idt_alpha.requires_grad = False
            self.loss_idt_A += self.criterionIdt(self.idt_Alpha, idt_alpha) * self.opt.lambda_edge * self.opt.lambda_identity * self.GAN_edge_weight
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        if self.multi_scale:
            # discriminator image domain
            self.loss_G_A  = self.netD_A.calc_gen_loss(self.fake_B)
            # discriminator edge domain
            self.loss_G_A += self.netD_C.calc_gen_loss(self.fake_NGF_A)
        else:
            # discriminator image domain
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            # discriminator edge domain
            if self.epsilon_multi_scale:
                self.loss_G_A += self.netD_C.calc_gen_loss(self.fake_NGF_A)
            else:
                self.loss_G_A += self.criterionGAN(self.netD_C(self.fake_NGF_A), True) * self.GAN_edge_weight

        #NGF loss
        self.loss_NGF_A = NGF_loss(self, self.fake_B, self.real_A, self.ada_epsitlon.detach() if self.detach else self.ada_epsitlon) * lambda_edge
        self.loss_G = self.loss_G_A+  self.loss_NGF_A + self.loss_idt_A
        self.loss_G.backward()


    def optimize_parameters(self):
        self.netG_A.cuda()
        self.netD_C.cuda()
        self.netD_A.cuda()
        self.forward()   
        self.set_requires_grad([self.netD_A, self.netD_C], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
                    # calculate gradients for G_A and G_B
        self.backward_G()
        self.set_requires_grad([self.netD_A, self.netD_C], True)
        self.optimizer_G.step()
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        if self.GAN_edge_weight > 0:
            self.optimizer_D_C.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
        if self.GAN_edge_weight > 0:
            self.optimizer_D_C.step()  # update D_A and D_B's weights