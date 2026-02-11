import torch
#from externals.TEED.ted_3D_no_maxpool import TED_3D 
from externals.TEED.ted_2D3D_no_maxpool import TED
from ourmodels.networks.bases import MONAI_UNet_feature
from utils.options import Options
import yaml 
cfg_path = '/home/ids/kgiraldo/These/configs/contours_config_3D_.yaml'
with open(cfg_path) as stream:#
    config = yaml.safe_load(stream)
opt_ = Options(**config['options'])
model_unet_3d = MONAI_UNet_feature(in_channels = opt_.in_channels, out_channels = opt_.out_channels, 
                spatial_dims = 3, num_res_units= opt_.num_res_units_G, 
                channels=opt_.U_channels, strides=opt_.U_strides, norm=opt_.norm).cuda()
cfg_path = '/home/ids/kgiraldo/These/configs/contours_config_2D.yaml'
with open(cfg_path) as stream:#
    config = yaml.safe_load(stream)
opt_ = Options(**config['options'])
model_unet_2d = MONAI_UNet_feature(in_channels = opt_.in_channels, out_channels = opt_.out_channels, 
                spatial_dims = 2, num_res_units= opt_.num_res_units_G, 
                channels=opt_.U_channels, strides=opt_.U_strides, norm=opt_.norm).cuda()
model_unet_2d.unet.features = True
for param in model_unet_2d.parameters():
    param.requires_grad = False
model_unet_2d.eval()
model_unet_2d.unet.layer = 4
model_unet_3d.unet.features = True
for param in model_unet_3d.parameters():
    param.requires_grad = False
model_unet_3d.eval()
model_unet_3d.unet.layer = 2
model_unet = None


from kornia.filters import SpatialGradient, SpatialGradient3d
device = torch.device(f'cuda:{0}')
model_teed3D = TED(gray_scale = True, dims=3).to(device)
model_teed2D = TED(gray_scale = True, dims=2).to(device)
model_ted = None
#weight_path = '/home/ids/kgiraldo/These/results/contours_3D_nmp/ep43.pt'
#model_ted.train()




import nibabel as nib

def TEED_loss(model, fake_B, real_A, NGF = "L1"):
    if not model.teed_load_flag:
        global model_ted
        model.teed_load_flag = True
        if model.spatial_dims == 3:
            model_ted = model_teed3D
        else:
            model_ted = model_teed2D
        model_ted.load_state_dict(torch.load(model.teed_load_path)['model'])
        for param in model_ted.parameters():
            param.requires_grad = False
        model_ted.eval()
    model_ted.out_features = False
    if NGF == 'L1':
        with torch.no_grad():
            out_GT = model_ted(real_A)
        out_pred = model_ted(fake_B)
        loss = torch.nn.L1Loss(reduction = 'mean')(out_GT, out_pred)
    if NGF == 'L2':
        with torch.no_grad():
            out_GT = model_ted(real_A)
        out_pred = model_ted(fake_B)
        loss = torch.nn.MSELoss(reduction = 'mean')(out_GT, out_pred)
    elif 'NTF' in NGF:
        model_ted.f_out = int(NGF[-1])
        with torch.no_grad():
            out_GT = model_ted(real_A)
        out_pred = model_ted(fake_B)
        if 'weighted' in NGF:
            model_ted.f_out = 1
            with torch.no_grad():
                out_GT_1 = model_ted(real_A)
                out_GT_1 = (out_GT_1 - out_GT_1.min()) / (out_GT_1.max() - out_GT_1.min())
            loss = torch.mean(edge_loss(model, out_GT, out_pred, 0, not_reduction=True)*out_GT_1)
        else:
            loss = edge_loss(model, out_GT, out_pred, 0)

    return loss

def Perceptual_loss(model, fake_B, real_A, perceptual):
    if 'TEED' in perceptual:
        if not model.teed_load_flag:
            global model_ted
            model.teed_load_flag = True
            if model.spatial_dims == 3:
                model_ted = model_teed3D
            else:
                model_ted = model_teed2D
            model_ted.load_state_dict(torch.load(model.teed_load_path)['model'])
            for param in model_ted.parameters():
                param.requires_grad = False
            model_ted.eval()
        model_ted.out_features = True
        if 'TEED' in perceptual:
            model_ted.f_out = int(perceptual[-1])
            with torch.no_grad():
                out_GT = model_ted(real_A)
            out_pred = model_ted(fake_B)
            loss = torch.nn.MSELoss(reduction = 'mean')(out_GT, out_pred)
            return loss
    else:
        if not model.teed_load_flag:
            global model_unet
            model.teed_load_flag = True
            if model.spatial_dims == 3:
                model_unet = model_unet_3d
            else:
                model_unet = model_unet_2d
            model_unet.load_state_dict(torch.load(model.teed_load_path)['model'])
            for param in model_unet.parameters():
                param.requires_grad = False
            model_unet.eval()
            model_unet.unet.layer = model.perceptual_layer
        model_unet.out_features = True
        with torch.no_grad():
            out_GT = model_unet(real_A)
        out_pred = model_unet(fake_B)
        loss = torch.nn.MSELoss(reduction = 'mean')(out_GT, out_pred)
        return loss





def to_64_patches(image):
  patches = image.unfold(2, 64, 64).unfold(3, 64, 64).unfold(4, 64, 64)
  patches = patches.contiguous().view(
      8, 1, 64, 64, 64
  )
  patches = patches.contiguous().view(
      8, -1, 1
  )
  return patches

def PatchNCELoss(real_a, fake_b, tau=0.07):
    real_a_p = to_64_patches(real_a)
    fake_b_p = to_64_patches(fake_b)
    # batch size, channel size, and number of sample locations
    B, C, S = real_a_p.shape

    # calculate v * v+: BxSx1
    l_pos = (fake_b_p * real_a_p).sum(dim=1)[:, :, None]

    # calculate v * v-: BxSxS
    l_neg = torch.bmm(real_a_p.transpose(1, 2), fake_b_p)

    # The diagonal entries are not negatives. Remove them.
    identity_matrix = torch.eye(S)[None, :, :].cuda()
    l_neg.masked_fill_(identity_matrix > 0, -float('inf'))

    # calculate logits: (B)x(S)x(S+1)
    logits = torch.cat((l_pos, l_neg), dim=2) / tau

    # return PatchNCE loss
    predictions = logits.flatten(0, 1)
    targets = torch.zeros(B * S, dtype=torch.long).cuda()
    return torch.mean(cross_entropy_loss(predictions, targets))

def edge_loss(model, fake_B, real_A, alpha_NGF, NGF='NGF', not_reduction = False, alpha_pixel_wise=None):
    eps = 1e-8
    if model.spatial_dims == 2:
        SG = SpatialGradient
    else:
        SG = SpatialGradient3d
    grad_src = SG()(fake_B)
    grad_tgt = SG()(real_A)
    if 'alpha_pixel_wise' in NGF:
        alpha_pixel_wise = torch.nn.Sigmoid()(alpha_pixel_wise)
        gradmag_src = torch.sqrt(torch.sum(torch.pow(grad_src,2), dim=2)+alpha_pixel_wise**2)
        gradmag_tgt = torch.sqrt(torch.sum(torch.pow(grad_tgt,2), dim=2)+alpha_pixel_wise**2)
    else:
        gradmag_src = torch.sqrt(torch.sum(torch.pow(grad_src,2), dim=2)+alpha_NGF**2)
        gradmag_tgt = torch.sqrt(torch.sum(torch.pow(grad_tgt,2), dim=2)+alpha_NGF**2)
    if 'NGF_weighted_pixel_wise' == NGF:
        alpha_pixel_wise = torch.nn.Sigmoid()(alpha_pixel_wise)
        NGF_L = 1-1/2*torch.pow(torch.sum(grad_src*grad_tgt, dim=2) / ((gradmag_src + eps) * (gradmag_tgt + eps)), 2)*alpha_pixel_wise
    NGF_L = 1-1/2*torch.pow(torch.sum(grad_src*grad_tgt, dim=2) / ((gradmag_src + eps) * (gradmag_tgt + eps)), 2)
    if 'UNET' in NGF:
        with torch.no_grad():
            out_GT = model_unet(real_A)
        out_pred = model_unet(fake_B)
        if NGF == 'UNETNGF':
            grad_src = SG()(out_GT)
            grad_tgt = SG()(out_pred)
            gradmag_src = torch.sqrt(torch.sum(torch.pow(grad_src,2), dim=2)+alpha_NGF**2)
            gradmag_tgt = torch.sqrt(torch.sum(torch.pow(grad_tgt,2), dim=2)+alpha_NGF**2)
            NGF_L = 1-1/2*torch.pow(torch.sum(grad_src*grad_tgt, dim=2) / ((gradmag_src + eps) * (gradmag_tgt + eps)), 2)
            return torch.mean(NGF_L)
        else:
            lossl1 = torch.nn.L1Loss(reduction = 'mean')(out_GT, out_pred)
            return lossl1 + torch.mean(NGF_L)

    if not_reduction:
        return NGF_L
    if NGF == 'NGF' or NGF == 'NGF_alpha_pixel_wise' or 'NGF_weighted_pixel_wise' == NGF:
        NGFM = torch.mean(NGF_L)
    elif 'TEED_weighted' in NGF:
        with torch.no_grad():
            out_GT = model_ted(real_A)
            out_GT = (out_GT - out_GT.min()) / (out_GT.max() - out_GT.min())
        NGFM = torch.mean(NGF_L*out_GT)
    elif 'scalar_TEED' in NGF:
        with torch.no_grad():
            out_GT = model_ted(real_A)
            out_GT = (out_GT - out_GT.min()) / (out_GT.max() - out_GT.min())
        out_pred = model_ted(fake_B)
        out_pred = (out_pred - out_pred.min()) / (out_pred.max() - out_pred.min())

        NGF_L = 1-1/2*torch.pow(torch.sum(out_GT*grad_src*grad_tgt*out_pred, dim=2) / ((gradmag_src + eps) * (gradmag_tgt + eps)), 2)
        NGFM = torch.mean(NGF_L)

    return NGFM
