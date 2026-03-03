import sys
import utils.data_loaders
from utils.optionsDL import Options
from utils.train_options import TrainOptions
# from logger import *
from utils.loops_train import train_loop, val_loop, metrics_segmentation, metrics_generator
import pandas as pd
import time
from models import create_model
import yaml 
from tqdm import tqdm
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.exposure import match_histograms
import imageio.v2 as imageio
import torch
import os
import numpy as np
from monai.inferers import SlidingWindowInferer, SliceInferer
import nibabel as nib
from kornia.filters import SpatialGradient, SpatialGradient3d
import torch.nn.functional as F
from monai.metrics import DiceMetric

# dice function used for training when validation is done during training
def dice_safe(pred, gt, eps=1e-8):
    pred = (pred > 0.5).float()
    gt   = (gt > 0.5).float()

    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()

    if union == 0:
        return torch.tensor(1.0, device=pred.device)

    return (2. * inter + eps) / (union + eps)


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



if __name__ == '__main__':

    # loading Cycle GAN old options 
    opt = TrainOptions().parse()
    # loading data loaders options
    with open(opt.cfg) as stream:
        config = yaml.safe_load(stream)
    opt_cfg = Options(**config['options'])
    opt.batch_size = opt_cfg.batch_size
    # creating dataloaders
    data_loader_class  = eval(f'utils.data_loaders.{config["data_loaders"]["imports"]["dataloader"]}')
    train_loader, val_loader, test_loader = data_loader_class(opt_cfg, config['data_loaders'], test_infer = opt.test)


    # creation of the model
    model = create_model(opt)
    # this pipeline accepts 2D and 3D models
    model.dim_ = opt.dim
    model.setup(opt)
    # if there is an genetor_A i the output folder, then load checkpiunt
    if (Path(opt.checkpoints_dir) / opt.name / 'latest_net_G_A.pth').exists():
        model.load_networks('latest')
    # initializing validation dict
    val_dict = {'epoch' : 0, 'best_psnr' : 0}
    # loading validation dict
    if (Path(opt.checkpoints_dir) / opt.name / 'dicts.pt').exists():
        val_dict = torch.load(Path(opt.checkpoints_dir) / opt.name / 'dicts.pt', weights_only=False)
        print(val_dict)
        val_dict['epoch'] += 1
    # at each epoch, images from the validation set will be created, in order to supervise training
    img_path = Path(opt.checkpoints_dir) / opt.name / 'images'
    if not img_path.exists():
        os.mkdir(img_path)
    # at test time, inference results will be saved in the inf folder
    inf_path = Path(opt.checkpoints_dir) / opt.name / 'inf'
    if not inf_path.exists():
        os.mkdir(inf_path)

    # in all models, the generator mapping is done through netG_A
    gen = model.netG_A
    gpu_device = torch.device(f'cuda:{0}')

    model.outdir = Path(opt.checkpoints_dir) / opt.name
    model.load_get_losses()
    total_time = 0
    if  not opt.test:
        print('Initiating training')
        epoch_0 = val_dict['epoch']
        for epoch in range(epoch_0, opt.n_epochs + opt.n_epochs_decay):
            print(f'Initiating training Epoch {epoch}')
            model.epoch = epoch
            model.init_get_losses()
            epoch_start_time = time.time()

            # getting data loader
            data_loop = tqdm(train_loader)
            data_loop.set_description(f"Epoch [{epoch}/{opt.n_epochs + opt.n_epochs_decay + 1}]")
            # putting all models in gpu_device
            for name in model.model_names:
                if 'optimizer' or 'scheduler' in name:
                    continue
                if isinstance(name, str):
                    getattr(model, 'net' + name).to(gpu_device)
            # putting in train mode
            gen.to(gpu_device)
            gen.train()

            for i, data in enumerate(data_loop):
                if epoch == epoch_0 and i == 0 and opt.model == 'cut': 
                    model.data_dependent_initialize(data)
                if opt.model == 'cut':
                    model.netF.to(gpu_device)
                    model.netD.to(gpu_device)
                # used to calculate batch computational time if opt.timer is on
                iter_start_time = time.time()
                
                #training 
                model.set_input(data)
                model.optimize_parameters()
                model.get_losses()
                #end training
                if opt.timer:
                    total_time += time.time() - iter_start_time
                    if i == 99:
                        print(total_time/100)
                        exit()

            model.plot_get_losses()

            #starting validation   
            data_loop = tqdm(val_loader)
            data_loop.set_description(f"Epoch [{epoch}/{opt.n_epochs + opt.n_epochs_decay + 1}]")     
            #putting model in eval mode
            gen.eval()
            # will be used to put the metric a each batch
            current_metric = []
            with torch.no_grad():
                for i, data in enumerate(data_loop):
                    # if the image is mostly background ignore for validation
                    if torch.sum(data["TGT"].to(gpu_device) != -1) < 40:
                        continue
                    # if the image is in 2D or 3D
                    if opt.dim == 2:
                        real_A = data["SRC"].to(gpu_device)[:,:,:,:,0]
                        real_B = data['TGT'].to(gpu_device)[:,:,:,:,0]
                    else:
                        real_A = data["SRC"].permute((0,1,4,2,3)).to(gpu_device)
                        real_B = data['TGT'].permute((0,1,4,2,3)).to(gpu_device)
                    # UNIT uses more than just G_A to pass from A to B domain
                    if 'UNIT' == opt.model:
                        model.netG_A.eval()
                        model.netG_B.eval()
                        h_a, _ = model.netG_A.encode(real_A)
                        fake_B = model.netG_B.decode(h_a)
                    # for the AdaNGF model
                    elif 'AdaNGF' in opt.model:
                        fake_B, alpha_B = gen(real_A, outA_bool=True)
                        NGF_fake = NGF_magnitude(model, real_A, alpha = alpha_B)
                        fake_NGFB = NGF_magnitude(model, real_B, alpha = model.epsilonT)
                    else:
                        fake_B = gen(real_A)
                    real_B_ = real_B[0,0].detach().cpu().numpy()
                    fake_B_ = fake_B[0,0].detach().cpu().numpy()
                    # is PSNR using histogram matching or not ignoring background
                    if opt.HM:
                        current_psnr = psnr(match_histograms(fake_B_[real_B_!=-1], real_B_[real_B_!=-1]), real_B_[real_B_!=-1], data_range = 2)
                    else:
                        current_psnr = psnr(fake_B_[real_B_!=-1], real_B_[real_B_!=-1], data_range = 2)
                    current_metric.append(current_psnr)

                    # calculating images to plot after validation
                    fake_B_val_0255 = np.uint8(np.clip(255*(fake_B[0,0,].cpu().detach().numpy().squeeze()+1)/2, 0,255))
                    real_B_val_0255 = np.uint8(np.clip(255*(real_B[0,0,].cpu().detach().numpy().squeeze()+1)/2, 0,255))
                    real_A_val_0255 = np.uint8(np.clip(255*(real_A[0,0,].cpu().detach().numpy().squeeze()+1)/2, 0,255))
                    if 'AdaNGF' in opt.model :
                        alpha_B_val_0255 = np.uint8(np.clip(255*(alpha_B[0,0,].cpu().detach().numpy().squeeze()), 0,255))
                        NGF_fake_val_0255 = np.uint8(np.clip(255*(NGF_fake[0,0,].cpu().detach().numpy().squeeze()), 0,255))
                    
                    if opt.dim == 3:
                        fake_B_val_0255 = fake_B_val_0255[fake_B_val_0255.shape[0]//2]
                        real_B_val_0255 = real_B_val_0255[real_B_val_0255.shape[0]//2]
                        real_A_val_0255 = real_A_val_0255[real_A_val_0255.shape[0]//2]
                        if 'AdaNGF' in opt.model:
                            alpha_B_val_0255 = alpha_B_val_0255[alpha_B_val_0255.shape[0]//2]
                            NGF_fake_val_0255 = NGF_fake_val_0255[NGF_fake_val_0255.shape[0]//2]
                    # getting current image anme  
                    fprefix = Path(data['SRC_meta_dict']['filename_or_obj'][0]).name
                    if i % opt_cfg.NUM_SAMPLES_MAX == 0:
                        fname_out = img_path  / (fprefix+'_fake_B_e%.3d.png' % epoch)
                        imageio.imwrite(fname_out, fake_B_val_0255)
                        if 'AdaNGF' in opt.model:
                            #saving alpha
                            #fname_out = img_path  / (fprefix+'_fake_B_e%.3d_alpha.png' % epoch)
                            #imageio.imwrite(fname_out, alpha_B_val_0255)
                            
                            #saving adapted NGF magnitude
                            fname_out = img_path  / (fprefix+'_fake_B_e%.3d_NGF.png' % epoch)
                            imageio.imwrite(fname_out, NGF_fake_val_0255)
                        # writing images
                        fname_out = img_path / (fprefix+'_real_B.png')
                        imageio.imwrite(fname_out, real_B_val_0255)
                        fname_out = img_path / (fprefix+'_real_A.png')
                        imageio.imwrite(fname_out, real_A_val_0255)
            # getting current PSNR
            psnr_epoch = np.mean(current_metric)
            print(f"Epoch {val_dict['epoch']}, PSNR {psnr_epoch}")
            val_dict['epoch'] += 1
            if psnr_epoch >= val_dict['best_psnr']:
                print('Saving Best!')
                val_dict['best_psnr'] = psnr_epoch
                model.save_networks('best')

            print('saving the latest model at the end of epoch %d' %
                (epoch))
            model.save_networks('latest')
            torch.save(val_dict, Path(opt.checkpoints_dir) / opt.name / 'dicts.pt')

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.n_epochs + opt.n_epochs_decay + 1, time.time() - epoch_start_time))
            model.update_learning_rate()
            
    else:
        print('Test / inference')
        # deciding to use last or best model
        if opt.last:
            model.load_networks('latest')
        else:
            model.load_networks('best')
            
        #putting model in test model
        gen.eval()
        data_loop = tqdm(test_loader)
        with torch.no_grad():
            # only necessary if the model will be test in other axis
            permute = [[(0,1,4,2,3), (1,2,0), 'Axial'],
                        [(0,1,2,4,3), (0,2,1), 'Sagittal'],
                        [(0,1,3,2,4), (1,0,2), 'Coronal'],
                        [(0,1,3,2,4), (1,0,2), 'Total']]
            if not opt.extra_view:
                permute[0][2] = opt.name
            dicts_out = [{},{},{},{},{},{},{},{}]
            for j in range(8):
                dicts_out[j]['File'] = []
            for i, data in enumerate(data_loop):
                out_list = []
                for perind in range(1+2*opt.extra_view):
                    # getting source and target images
                    real_A = data["SRC"].to(gpu_device)
                    real_B = data["TGT"].to(gpu_device)
                    real_A = real_A.permute(permute[perind][0])
                    # getting output file name
                    fname_ref = Path(data["TGT_meta_dict"]['filename_or_obj'][0])
                    fname_out = Path(opt.checkpoints_dir) / opt.name / 'inf' /fname_ref.name
                    # adding output file to the metrics
                    dicts_out[perind*2]['File'].append(fname_ref.name)
                    dicts_out[perind*2+1]['File'].append(fname_ref.name)
                    if opt.extra_view and perind == 0:
                        dicts_out[6]['File'].append(fname_ref.name)
                        dicts_out[7]['File'].append(fname_ref.name)
                    
                    # defining window inferer using monai
                    if opt.dim==3:
                        window_inferer = SlidingWindowInferer(**config["data_loaders"]['SlidingWindowInferer'], mode = 'gaussian')
                    else:
                        window_inferer = SliceInferer(**config["data_loaders"]['SlidingWindowInferer'], mode = 'gaussian')
                    
                    # definining unit eval mode
                    if 'UNIT' == opt.model:
                        def wrapped_model(input_slice):
                            h_a, _ = model.netG_A.encode(input_slice)
                            fake_B = model.netG_B.decode(h_a)
                            return fake_B
                        model.netG_A.eval()
                        model.netG_B.eval()
                        output = window_inferer(inputs=real_A, network=wrapped_model)
                    # definining AdaNGF eval mode
                    elif 'AdaNGF' in opt.model:
                        def wrapped_model(input_slice):
                            # each patch output both image domain and NGF magnitude domain
                            out1, out2 = gen(input_slice, outA_bool=True)
                            out3 = torch.concat((out1, out2), dim=1)
                            return out3
                        output = window_inferer(inputs=real_A, network=wrapped_model)
                        output_A = output[0,1].permute(permute[perind][1]).detach().cpu().numpy()
                        
                        if model.dim_ == 2:
                            # NGF magnitude of source and target calculated with 
                            NGF_real_A = NGF_magnitude(model, real_A[0], alpha = model.epsilonT)[0].permute(permute[perind][1]).detach().cpu().numpy()
                            NGF_output = NGF_magnitude(model, real_A[0], alpha = output[0,[1]])[0].permute(permute[perind][1]).detach().cpu().numpy()
                            NGF_real_B = NGF_magnitude(model, real_B.permute(permute[perind][0])[0], alpha =  model.epsilonT)[0].permute(permute[perind][1]).detach().cpu().numpy()
                        else:
                                NGF_real_A = NGF_magnitude(model, real_A, alpha = model.epsilonT)[0,0].permute(permute[perind][1]).detach().cpu().numpy()
                                NGF_output = NGF_magnitude(model, real_A, alpha = output[[0],[1]])[0,0].permute(permute[perind][1]).detach().cpu().numpy()
                                NGF_real_B = NGF_magnitude(model, real_B.permute(permute[perind][0]), alpha =  model.epsilonT)[0,0].permute(permute[perind][1]).detach().cpu().numpy()
                    else:
                        # simple eval mode
                        def wrapped_model(input_slice):
                            out1= gen(input_slice)
                            return out1
                        output = window_inferer(inputs=real_A, network=wrapped_model)
                    #getting output / infered image
                    output = output[0,0].permute(permute[perind][1]).detach().cpu().numpy()
                    output = np.clip(output, a_min=-1, a_max=1)
                output = output.copy()
                out_list.append(output.copy())
                # source and target in numpy arrays
                real_A_ = real_A[0,0].permute(permute[perind][1]).detach().cpu().numpy().copy()
                real_B_ = real_B[0,0].detach().cpu().numpy().copy()
                # nibabel affine and header for nibabel outpout
                nib_real = nib.load(fname_ref)
                output_masked = output.copy()
                output_masked[real_B_ == -1] = -1
                nib.save(nib.Nifti1Image(output_masked, nib_real.affine, nib_real.header), str(fname_out).replace('sCenter.nii', permute[perind][2] + '.nii'))
                
                if opt.harmonisation:
                    output[real_B_== -1] = -1
                    nib.save(nib.Nifti1Image(output, nib_real.affine, nib_real.header), str(fname_out).replace('sCenter.nii', permute[perind][2] + '.nii'))
                
                # put background in GT as background in generated
                output[real_B_<= (-1+10e-5)] = -1
                if 'AdaNGF' in opt.model:
                    # saving all NGF magnitude images
                    nib.save(nib.Nifti1Image(NGF_output, nib_real.affine, nib_real.header), str(fname_out).replace('_sCenter.nii', f'_{opt.name}_NGF_fake_source.nii'))
                    nib.save(nib.Nifti1Image(NGF_real_B, nib_real.affine, nib_real.header), str(fname_out).replace('_sCenter.nii', f'_{opt.name}_NGF_real_target.nii'))
                    nib.save(nib.Nifti1Image(NGF_real_A, nib_real.affine, nib_real.header), str(fname_out).replace('_sCenter.nii', f'_{opt.name}_NGF_real_source.nii'))
                if not opt.skip_metrics:
                    # image translation metrics   
                    metrics_generator(real_B_, output.copy(), dicts_out[perind*2], real_B_ != -1000, mask = True)
                    print(dicts_out[perind*2]['PSNR'][-1])
                # if histogram matching is necessary for output images
                if opt.HM:
                    output[real_B_ != -1] = match_histograms(output[real_B_ != -1], real_B_[real_B_ != -1])
                    nib.save(nib.Nifti1Image(output, nib_real.affine, nib_real.header), str(fname_out).replace('sCenter.nii', permute[perind][2] + '_HM.nii'))
                    if not opt.skip_metrics: 
                        metrics_generator(real_B_, output.copy(), dicts_out[perind*2+1], real_B_ != -1000, mask = True)
                # saving images in other axis
                if 1+2*opt.extra_view == 3:
                    out_list = np.mean(out_list, axis=0)
                    nib.save(nib.Nifti1Image(out_list, nib_real.affine, nib_real.header), str(fname_out).replace('sCenter.nii','Total.nii'))
                    out_list[real_B_== -1] = -1
                    metrics_generator(real_B_, out_list.copy(), dicts_out[6], real_B_ != -1000, mask = True)
                    out_list[real_B_ != -1] = match_histograms(out_list[real_B_ != -1], real_B_[real_B_ != -1])
                    nib.save(nib.Nifti1Image(out_list, nib_real.affine, nib_real.header), str(fname_out).replace('sCenter.nii', 'Total_HM.nii'))
                    metrics_generator(real_B_, out_list.copy(), dicts_out[7], real_B_ != -1000, mask = True)

            # saving results as a csv in inf dir
            for i in range(1 + 3*opt.extra_view ):
                for key_ in dicts_out[i*2].keys():
                    if key_ == 'File':
                        dicts_out[i*2][key_].append('mean')
                        dicts_out[i*2][key_].append('STD')
                    else:
                        dicts_out[i*2][key_].append(np.mean(dicts_out[i*2][key_]))
                        dicts_out[i*2][key_].append(np.std(dicts_out[i*2][key_][:-1]))
                pd.DataFrame.from_dict(dicts_out[i*2]).to_csv(inf_path / f'metrics_{permute[i][2]}.csv', index=False)
                for key_ in dicts_out[i*2+1].keys():
                    if key_ == 'File':
                        dicts_out[i*2+1][key_].append('mean')
                        dicts_out[i*2+1][key_].append('STD')
                    else:
                        dicts_out[i*2+1][key_].append(np.mean(dicts_out[i*2+1][key_]))
                        dicts_out[i*2+1][key_].append(np.std(dicts_out[i*2+1][key_][:-1]))
                pd.DataFrame.from_dict(dicts_out[i*2+1]).to_csv(inf_path / f'metrics_{permute[i][2]}_HM.csv', index=False)








