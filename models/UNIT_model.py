# This code is released under the CC BY-SA 4.0 license.

import torch
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import networks_UNIT
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

class UNITModel(BaseModel):
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
        parser.add_argument('--disc', type=bool, default=False, help='weight for cycle loss (A -> B -> A)')
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        '''
        gen:
        dim: 64                     # number of filters in the bottommost layer
        activ: relu                 # activation function [relu/lrelu/prelu/selu/tanh]
        n_downsample: 2             # number of downsampling layers in content encoder
        n_res: 4                    # number of residual blocks in content encoder/decoder
        pad_type: reflect           # padding type [zero/reflect]
        dis:
        dim: 64                     # number of filters in the bottommost layer
        norm: none                  # normalization layer [none/bn/in/ln]
        activ: lrelu                # activation function [relu/lrelu/prelu/selu/tanh]
        n_layer: 4                  # number of layers in D
        gan_type: lsgan             # GAN loss [lsgan/nsgan]
        num_scales: 3               # number of scales
        pad_type: reflect           # padding type [zero/reflect]
        '''
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.alpha_NGF = opt.alpha_NGF
        self.disc = opt.disc
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = []
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')

        self.visual_names = visual_names_A # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'D_A', 'G_B', 'D_B', 'optimizer_D', 'optimizer_G', 'schedulers']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks_UNIT.VAEGen(1, {'dim' : 64, 'activ' : 'relu', 'n_downsample' : 2, 'n_res' : 4, 'pad_type' : 'reflect'})
        self.netG_A.cuda()
        self.netG_B = networks_UNIT.VAEGen(1, {'dim' : 64, 'activ' : 'relu', 'n_downsample' : 2, 'n_res' : 4, 'pad_type' : 'reflect'})  # auto-encoder for domain b
        self.netG_B.cuda()
        if self.isTrain:  # define discriminators                 # number of filters in the bottommost layer
            if self.disc:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, dim = opt.dim)
                self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, dim = opt.dim)
            else:
                self.netD_A = networks_UNIT.MsImageDis(1, {'dim':64, 'norm': 'none', 'activ': 'lrelu', 'n_layer':4, 'gan_type':'lsgan', 'num_scales':3, 'pad_type':'reflect'})  # discriminator for domain a
                self.netD_A.cuda()
                self.netD_B = networks_UNIT.MsImageDis(1, {'dim':64, 'norm': 'none', 'activ': 'lrelu', 'n_layer':4, 'gan_type':'lsgan', 'num_scales':3, 'pad_type':'reflect'})  # discriminator for domain b
                self.netD_B.cuda()

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
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
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
        if np.random.random() > 0.7 or True:
            self.h_a, self.n_a = self.netG_A.encode(self.real_A)
            self.h_b, self.n_b = self.netG_B.encode(self.real_B)
            # decode (within domain)
            self.x_a_recon = self.netG_A.decode(self.h_a + self.n_a)
            self.x_b_recon = self.netG_B.decode(self.h_b + self.n_b)
            # decode (cross domain)
            self.x_ba = self.netG_A.decode(self.h_b + self.n_b)
            self.x_ab = self.netG_B.decode(self.h_a + self.n_a)
            # encode again
            self.h_b_recon, self.n_b_recon = self.netG_A.encode(self.x_ba)
            self.h_a_recon, self.n_a_recon = self.netG_B.encode(self.x_ab)
            # decode again (if needed)
            self.x_aba = self.netG_A.decode(self.h_a_recon + self.n_a_recon)
            self.x_bab = self.netG_B.decode(self.h_b_recon + self.n_b_recon)
        else:
            if np.random.random() > 0.5:
                if self.dim_ == 3:
                    with torch.no_grad():
                        b, c, h, w, d = self.real_A.shape
                        img_input = torch.zeros_like(self.real_A) -1 
                        img_output = torch.zeros_like(self.real_B) - 1
                        h2 = np.random.randint(h//4, h//2)
                        w2 = np.random.randint(w//4, w//2)
                        d2 = np.random.randint(d//4, d//2)
                        img_input[:, :, h//2 - h2:h//2 + h2, w//2 - w2:w//2 + w2, d//2 - d2:d//2 + d2] = self.real_A[:, :, h//2 - h2:h//2 + h2, w//2 - w2:w//2 + w2, d//2 - d2:d//2 + d2]
                        img_output[:, :, h//2 - h2:h//2 + h2, w//2 - w2:w//2 + w2, d//2 - d2:d//2 + d2] = self.real_B[:, :, h//2 - h2:h//2 + h2, w//2 - w2:w//2 + w2, d//2 - d2:d//2 + d2]
                        self.real_A = img_input.detach()
                        self.real_B = img_output.detach()
                      # G_A(A)
                else:
                    with torch.no_grad():
                        b, c, h, w = self.real_A.shape
                        img_input = torch.zeros_like(self.real_A) -1 
                        img_output = torch.zeros_like(self.real_B) - 1
                        h2 = np.random.randint(h//4, h//2)
                        w2 = np.random.randint(w//4, w//2)
                        img_input[:, :, h//2 - h2:h//2 + h2, w//2 - w2:w//2 + w2] = self.real_A[:, :, h//2 - h2:h//2 + h2, w//2 - w2:w//2 + w2]
                        img_output[:, :, h//2 - h2:h//2 + h2, w//2 - w2:w//2 + w2] = self.real_B[:, :, h//2 - h2:h//2 + h2, w//2 - w2:w//2 + w2]
                        self.real_A = img_input.detach()
                        self.real_B = img_output.detach()
            self.real_A_ = self.real_A.clone().detach()
            if np.random.random() > 0.5:
                with torch.no_grad():
                    if self.dim_ == 3:
                        r = R.from_rotvec([(np.random.random()-0.5)*np.pi*2/9, (np.random.random()-0.5)*np.pi*2/9, (np.random.random()-0.5)*np.pi*2/9])
                        self.real_A = rotate(self.real_A_, torch.tensor(r.as_matrix()).cuda()).detach()
                        self.real_A_ = self.real_A.clone().detach()
                        self.real_B = rotate(self.real_B, torch.tensor(r.as_matrix()).cuda()).detach()
                    else:
                        angle = (np.random.random()-0.5)*np.pi*2/9
                        self.real_A = torchvision.transforms.functional.rotate(self.real_A_, angle).detach()
                        self.real_A_ = self.real_A.clone().detach()
                        self.real_B = torchvision.transforms.functional.rotate(self.real_B, angle).detach()
            if np.random.random() > 0.5:
                with torch.no_grad():
                    if self.dim_ == 3:
                        b, _, d, h, w  = self.real_A.shape
                        device_ = self.real_A.device
                        w2 = np.random.randint(w)//2
                        h2 = np.random.randint(h)//2
                        d2 = np.random.randint(d)//2
                        locations_x = torch.linspace(0, w-1, w).view(1, 1, 1, w).to(device_).expand(b, d, h, w) + w2 - w//4
                        locations_y = torch.linspace(0, h-1, h).view(1, 1, h, 1).to(device_).expand(b, d, h, w) + h2 - h//4
                        locations_z = torch.linspace(0, d-1,d).view(1, d, 1, 1).to(device_).expand(b, d, h, w) + d2 - d//4
                        # stack locations
                        grid = torch.stack([(locations_z - d//2)*2/d, (locations_y - h//2)*2/h, (locations_x - w//2)*2/w,], dim=4).view(b, d, h, w, 3)
                        real_B = F.grid_sample(input=self.real_B+2, grid=grid, mode='bilinear',  align_corners=True)
                        real_B[real_B <= 1] = 1
                        real_B = real_B - 2
                        self.real_B = real_B.detach()
                        real_A = F.grid_sample(input=self.real_A+2, grid=grid, mode='bilinear',  align_corners=True)
                        real_A[real_A <= 1] = 1
                        real_A = real_A - 2
                        self.real_A = real_A.detach()
                        self.real_A_ = self.real_A.clone().detach()
                    else:
                        b, _, h, w  = self.real_A.shape
                        device_ = self.real_A.device
                        w2 = np.random.randint(w)
                        h2 = np.random.randint(h)
                        locations_x = torch.linspace(0, w-1, w).view(1, 1, w).to(device_).expand(b, h, w) + w2 - w//2
                        locations_y = torch.linspace(0, h-1, h).view(1, h, 1).to(device_).expand(b, h, w) + h2 - h//2
                        # stack locations
                        grid = torch.stack([(locations_y - h//2)*2/h, (locations_x - w//2)*2/w], dim=3).view(b, h, w, 2)
                        real_B = F.grid_sample(input=self.real_B+2, grid=grid, mode='bilinear',  align_corners=True)
                        real_B[real_B <= 1] = 1
                        real_B = real_B - 2
                        self.real_B = real_B.detach()
                        real_A = F.grid_sample(input=self.real_A+2, grid=grid, mode='bilinear',  align_corners=True)
                        real_A[real_A <= 1] = 1
                        real_A = real_A - 2
                        self.real_A = real_A.detach()
                        self.real_A_ = self.real_A.clone().detach()
        '''out_path = Path('/home/ids/kgiraldo/These/results/SynthRad25/PIX2PIX3d_new_196/experiments')
        fname_ref = Path(self.input_["TGT_meta_dict"]['filename_or_obj'][0])
        fname_out = fname_ref.name
        nib_real = nib.load(fname_ref)
        nib.save(nib.Nifti1Image(np.hstack([self.real_B[0,0].detach().cpu().numpy(),self.real_A[0,0].detach().cpu().numpy()]), nib_real.affine, nib_real.header), out_path / fname_out)'''
        #self.fake_B = self.netG_A(self.real_A)  # G_A(A)

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
        x_ba = self.fake_B_pool.query(self.x_ba)
        x_ab = self.fake_A_pool.query(self.x_ab)
        if self.disc:
            pred_real = self.netD_A(self.real_A)
            loss_D_real = self.criterionGAN(pred_real, True)
            pred_fake = self.netD_A(x_ba.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5


            pred_real = self.netD_B(self.real_B)
            loss_D_real = self.criterionGAN(pred_real, True)
            pred_fake = self.netD_B(x_ab.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D += (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
        else:
            
            self.loss_dis_a = self.netD_A.calc_dis_loss(x_ba.detach(), self.real_A)
            self.loss_dis_b = self.netD_B.calc_dis_loss(x_ab.detach(), self.real_B)
            self.loss_dis_total = self.loss_dis_a + self.loss_dis_b
            self.loss_dis_total.backward()
    
    
    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))
    
    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss
    
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        self.loss_gen_recon_x_a = self.recon_criterion(self.x_a_recon, self.real_A) * 10
        self.loss_gen_recon_x_b = self.recon_criterion(self.x_b_recon, self.real_B) * 10
        self.loss_gen_recon_kl_a = self.__compute_kl(self.h_a) * 0.01 
        self.loss_gen_recon_kl_b = self.__compute_kl(self.h_b) * 0.01 
        self.loss_gen_cyc_x_a = self.recon_criterion(self.x_aba, self.real_A) * 10
        self.loss_gen_cyc_x_b = self.recon_criterion(self.x_bab, self.real_B) * 10
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(self.h_a_recon) * 0.01
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(self.h_b_recon) * 0.01
        # GAN loss
        if self.disc:
            self.loss_gen_adv_a = self.criterionGAN(self.netD_A(self.x_ba), True)
            self.loss_gen_adv_b = self.criterionGAN(self.netD_B(self.x_ab), True)
        else:
            self.loss_gen_adv_a = self.netD_A.calc_gen_loss(self.x_ba)
            self.loss_gen_adv_b = self.netD_B.calc_gen_loss(self.x_ab)


        self.loss_G = self.loss_gen_adv_b + self.loss_gen_adv_a + self.loss_gen_recon_kl_cyc_bab + self.loss_gen_recon_kl_cyc_aba + self.loss_gen_cyc_x_a + self.loss_gen_cyc_x_b + self.loss_gen_recon_kl_b + self.loss_gen_recon_kl_a + self.loss_gen_recon_x_a + self.loss_gen_recon_x_b
        self.loss_G.backward()

    def optimize_parameters(self):
        self.netG_B.cuda()
        self.netG_A.cuda()
        self.netG_B.train()
        self.netG_A.cuda()
        self.netD_B.cuda()
        self.netD_A.cuda()
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
