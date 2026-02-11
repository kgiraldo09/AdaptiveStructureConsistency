from pathlib import Path
from monai.networks.layers.factories import Act, Norm

class Options():
    def __init__(self, in_channels = 1, out_channels = 1, num_filters_d = 128, num_layers_d = 4,
                 num_d = 3, num_res_units_G = 18, spatial_dims = 3, U_channels=(64, 128, 256, 512, 512),
                 U_strides=(2, 2, 2, 2), kernel_size=3, N_train = 500, learning_rate = 2e-4,
                 lambda_identity = 0, lambda_gan = 1, lambda_NGF = 3, alpha_NGF = 0.1, num_epochs = 100,
                 lambda_l1 = 0, input_dir = '', input_dir_processed = '', results_dir = '', EXPERIMENT_PREFIX = '',
                 batch_size = 1, generator=True, load_model = '', warming_epochs=0, test=False,
                 IO_MODEL = {}, retraining_epoch = 1000, lambda_identity_ouput = 0, norm='Norm.INSTANCE',
                 NUM_SAMPLES_MAX = 8, lambda_NCE = 0, NoTranspose=False, structural_loss='NGF', lambda_perceptual = 0, perceptual_loss = 'TEEED_1',
                 TEED_weihgts = '/home/ids/kgiraldo/These/results/contours_3D_nmp/ep43.pt', multiplicative = False, perceptual_layer = 3,
                 HM = False, lambda_cycle = 0, paired = False, HR = False, ADAS_seg=False):
        self.ADAS_seg = ADAS_seg
        self.HR = HR
        self.paired = paired
        self.lambda_cycle = lambda_cycle
        self.HM = HM
        self.perceptual_layer = perceptual_layer
        self.multiplicative = multiplicative
        self.TEED_weihgts = TEED_weihgts
        self.lambda_perceptual = lambda_perceptual
        self.perceptual_loss = perceptual_loss
        # model parameters
        self.structural_loss = structural_loss
        self.NoTranspose = NoTranspose
        self.NUM_SAMPLES_MAX = NUM_SAMPLES_MAX
        self.norm = eval(norm)
        self.retraining_epoch = retraining_epoch
        self.IO_MODEL = IO_MODEL
        self.warming_epochs = warming_epochs
        self.generator = generator
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters_d = num_filters_d  # number of filters in the discriminator
        self.num_layers_d = num_layers_d  # number of layers in the discriminator (i.e. the receptive field)
        self.num_d = num_d # number of discriminators (à la pix2pix HD)
        self.num_res_units_G = num_res_units_G #

        self.spatial_dims = spatial_dims
        self.U_channels=U_channels #(64, 128, 256, 512, 512, 512, 512)
        self.U_strides=U_strides#(2, 2, 2, 2, 2, 2)
        self.kernel_size=kernel_size

        # training parameters
        self.N_train = N_train # number of patient IDs for training (you need at least 2*N total as we perform unpaired training)
        self.num_epochs = num_epochs  # 300 is enough
        self.learning_rate = learning_rate  # typical value for CycleGAN
        self.lambda_identity = lambda_identity  # identity loss
        self.lambda_gan = lambda_gan # GAN loss weight
        self.lambda_NGF = lambda_NGF #
        self.alpha_NGF = alpha_NGF # best .15
        self.lambda_l1 = lambda_l1 #  # best 100 for pix2pix, no reconstruction loss for unpaired training
        self.lambda_identity_ouput = lambda_identity_ouput
        self.lambda_NCE = lambda_NCE

        # I/0 parametes
        self.test = test
        self.load_model = load_model
        self.batch_size = batch_size
        self.input_dir = Path(input_dir) if not isinstance(input_dir, list) else [Path(input_dir_) for input_dir_ in input_dir]  # original BraTS23 data folder
        self.input_dir_processed = Path(eval(input_dir_processed)) # where 2d images will be stored for training
        self.results_dir = Path(results_dir)
        #self.EXPERIMENT_PREFIX = Path(eval(EXPERIMENT_PREFIX))
