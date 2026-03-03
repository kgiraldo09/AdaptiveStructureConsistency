# This code is released under the CC BY-SA 4.0 license.

import torch
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from externals.cwdm_new.guided_diffusion.script_util import create_model_and_diffusion
from externals.cwdm_new.guided_diffusion.resample import create_named_schedule_sampler
from externals.cwdm_new.guided_diffusion import gaussian_diffusion as gd
from kornia.filters import SpatialGradient, SpatialGradient3d
import numpy as np

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

def edge_loss(model, fake_B, real_A, alpha_NGF):
    if model.dim_ == 2:
        SpatialGradient_ = SpatialGradient
    else:
        SpatialGradient_ = SpatialGradient3d
    eps = 1e-8
    grad_src = SpatialGradient_()(fake_B)
    grad_tgt = SpatialGradient_()(real_A)
    gradmag_src = torch.sqrt(torch.sum(torch.pow(grad_src,2), dim=2)+alpha_NGF**2)
    gradmag_tgt = torch.sqrt(torch.sum(torch.pow(grad_tgt,2), dim=2)+alpha_NGF**2)
    NGF_L = 1-1/2*torch.pow(torch.sum(grad_src*grad_tgt, dim=2) / ((gradmag_src + eps) * (gradmag_tgt + eps)), 2)
    NGF_L = 1-1/2*torch.pow(torch.sum(grad_src*grad_tgt, dim=2) / ((gradmag_src + eps) * (gradmag_tgt + eps)), 2)
    NGFM = torch.mean(NGF_L)
    return NGFM


def NGF_comnpute(model, img):
    if model.dim_ == 2:
        SG = SpatialGradient
    else:
        SG = SpatialGradient3d
    eps = 1e-8
    grad_img = SG()(img)
    gradmag_img = torch.sum(torch.pow(grad_img,2), dim=2)+eps
    NGF = grad_img * grad_img / gradmag_img[:,:,None]
    NGF = NGF.squeeze(1)
    return NGF



class CDMModel(BaseModel):
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
        parser.add_argument('--sqrt', type=bool, default=False, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--signed', type=bool, default=False, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--alpha_NGF', type=float, default=0.0000, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--concat_NGF', type=bool, default=False, help='weight for cycle loss (A -> B -> A)')
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.sqrt = opt.sqrt
        self.signed = opt.signed
        self.alpha_NGF = opt.alpha_NGF
        self.concat_NGF = opt.concat_NGF
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['L2_A', 'idt_A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')

        self.visual_names = visual_names_A # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        dict_model_and_diffusion = {'image_size': 128, 'num_channels': 64, 'num_res_blocks': 2, 
                                'num_heads': 1, 'num_heads_upsample': -1, 'num_head_channels': -1, 
                                'attention_resolutions': '', 'channel_mult': '1,2,2,4,4', 'dropout': 0.0, 
                                'class_cond': False, 'use_checkpoint': False, 'use_scale_shift_norm': False, 
                                'resblock_updown': True, 'use_fp16': False, 'use_new_attention_order': False, 
                                'dims': 3, 'num_groups': 4, 'in_channels': 4, 'out_channels': 1, 
                                'bottleneck_attention': False, 'resample_2d': False, 'additive_skips': False, 
                                'mode': 'i2i', 'use_freq': False, 'predict_xstart': True, 
                                'learn_sigma': False, 'diffusion_steps': 1000, 'noise_schedule': 'linear', 
                                'timestep_respacing': '', 'use_kl': False, 'rescale_timesteps': False, 
                                'rescale_learned_sigmas': False, 'dataset': 'brats'}
        betas = gd.get_named_beta_schedule(dict_model_and_diffusion['noise_schedule'], 
                                           dict_model_and_diffusion['diffusion_steps'])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)   
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        model, diffusion = create_model_and_diffusion(**dict_model_and_diffusion)
        self.netG_A = model
        self.netG_A.to(self.device)
        self.diffusion = diffusion
        self.schedule_sampler = create_named_schedule_sampler('uniform', diffusion,  maxt=1000)
        #t, weights = self.schedule_sampler.sample(1, 'cuda:0')
        #model.to('cuda:0')
        #target_model = torch.randn(1,1,224,224,224).to('cuda:0')
        #cond = torch.randn(1,3,224,224,224).to('cuda:0')
        #noise = torch.randn_like(target_model)
        #x_t = q_sample(target_model, t, noise=noise)
        #x_t = torch.cat([x_t, cond], dim=1) 
        #out_model = model(x_t)
        #print(out_model.shape)
        self.criterionL2  = torch.nn.MSELoss()#(out_model, target_model)
        #loss.backward()
        '''sample = diffusion.p_sample_loop(model=model,
                           shape=noise.shape,
                           noise=noise,
                           cond=cond,
                           clip_denoised=False,
                           model_kwargs={})'''
        #exit()
        #self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, dim = opt.dim)

        #if self.isTrain:  # define discriminators
        #    self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
        #                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, dim = opt.dim)

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
            self.optimizer_G = torch.optim.AdamW(self.netG_A.parameters(), lr=10e-5, weight_decay=0)
            #torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            #self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        if self.dim_ == 2: 
            #self.real_A = input['SRC'][:,:,:,:,0].to(self.device).float()
            self.real_A = NGF_comnpute(self, input['TGT'][:,:,:,:,0].to(self.device).float()).detach()
            self.real_B = input['TGT'][:,:,:,:,0].to(self.device).float()
        else:
            #self.real_A = input['SRC'].permute((0,1,4,2,3)).to(self.device).float()
            self.real_A = NGF_comnpute(self, input['TGT'].permute((0,1,4,2,3)).to(self.device).float()).detach()
            self.real_B = input['TGT'].permute((0,1,4,2,3)).to(self.device).float()
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        t, weights = self.schedule_sampler.sample(1, self.device)
        target_model = self.real_B
        cond = self.real_A
        noise = torch.randn_like(target_model)
        x_t = q_sample(self, target_model, t, noise=noise)
        x_t = torch.cat([x_t, cond], dim=1) 
        self.fake_B = self.netG_A(x_t, t)

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
        #self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        #self.loss_NGF_A = edge_loss(self, self.fake_B, self.real_A, 0.0001) * lambda_A
        self.loss_L2_A = self.criterionL2(self.fake_B, self.real_B) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_G = self.loss_L2_A + self.loss_idt_A
        self.loss_G.backward()
    def inference(self, inputNGF):
        print(inputNGF.shape)
        noise = torch.randn_like(inputNGF[:,[0]])
        fake_B = self.diffusion.p_sample_loop(model=self.netG_A,
                           shape=noise.shape,
                           noise=noise,
                           cond=inputNGF,
                           clip_denoised=False,
                           model_kwargs={})
        return fake_B

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        #self.set_requires_grad([self.netD_A], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        #self.set_requires_grad([self.netD_A], True)
        #self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        #self.backward_D_A()      # calculate gradients for D_A
        #self.optimizer_D.step()  # update D_A and D_B's weights
