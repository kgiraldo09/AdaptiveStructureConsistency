from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.exposure import match_histograms
import imageio.v2 as imageio
import numpy as np
from pathlib import Path
import gudhi as gd
from utils.metrics_ATM22 import precision_calculation, specificity_calculation, sensitivity_calculation, dice_coefficient_score_calculation, clDice
from monai.losses import SoftclDiceLoss
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils.Metrics_IQA import haar_psi
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=255, kernel_size = 7)
ms_ssim_t = MultiScaleStructuralSimilarityIndexMeasure(data_range=255, kernel_size = 3)


def get_generator_loss(opt, model, fake_B, pred_fake_B, real_B, real_A,loss_plot = None, pred_fake_A = None, cyc_B = None, cyc_A = None):
    generator_loss = 0.0

    if opt.lambda_l1_:
        if model.out_channels == 2:
            l1_loss = model.compute_l1_loss(fake_B[:,[0]], real_B)
        else:
            l1_loss = model.compute_l1_loss(fake_B, real_B)
        generator_loss += opt.lambda_l1_*l1_loss
        
    if opt.lambda_NCE:
        if model.out_channels == 2:
            nce_loss = model.compute_PatchNCELoss(real_A.requires_grad_(requires_grad=False), fake_B[:,0])
        else:
            nce_loss = model.compute_PatchNCELoss(real_A.requires_grad_(requires_grad=False), fake_B)
        generator_loss += opt.lambda_NCE*nce_loss
    if opt.lambda_cycle:
        cycle_loss = model.compute_cycleLoss(real_A, real_B, cyc_A, cyc_B)
        generator_loss += opt.lambda_cycle*cycle_loss
        loss_plot['Loss_Train']['cycle'][-1].append((opt.lambda_cycle*cycle_loss).item())
    if opt.lambda_NGF:
        if model.out_channels == 2:
            NGF_loss = model.compute_NGF_loss(fake_B[:,[0]], real_A, opt.alpha_NGF, NGF = opt.structural_loss, alpha_pixel_wise = fake_B[:,[1]])
        else:
            NGF_loss = model.compute_NGF_loss(fake_B, real_A, opt.alpha_NGF, NGF = opt.structural_loss, alpha_pixel_wise = fake_B)
        generator_loss += opt.lambda_NGF*NGF_loss
        loss_plot['Loss_Train']['NGF'][-1].append((opt.lambda_NGF*NGF_loss).item())
    if opt.lambda_gan:
        if model.model == 'cycle':
            adv_loss = model.compute_adv_loss(pred_fake_B, pred_fake_A)
        else:
            adv_loss = model.compute_adv_loss(pred_fake_B)
        generator_loss += opt.lambda_gan*adv_loss
        loss_plot['Loss_Train']['Gan'][-1].append((opt.lambda_gan*adv_loss).item())
    if opt.lambda_perceptual:
        if 'contrary' in opt.perceptual_loss:
            if model.out_channels == 2:
                perceptual_loss = model.compute_perceptual_loss(fake_B[:,[0]], real_B, opt.perceptual_loss)
            else:
                perceptual_loss = model.compute_perceptual_loss(fake_B, real_B, opt.perceptual_loss)
        else:
            if model.out_channels == 2:
                perceptual_loss = model.compute_perceptual_loss(fake_B[:,[0]], real_A, opt.perceptual_loss)
            else:
                perceptual_loss = model.compute_perceptual_loss(fake_B, real_A, opt.perceptual_loss)
        generator_loss += opt.lambda_perceptual*perceptual_loss
        loss_plot['Loss_Train']['Perceptual'][-1].append((opt.lambda_perceptual*perceptual_loss).item())
    if opt.lambda_identity:
        identity_loss = model.compute_identity_loss(real_B)
        generator_loss += opt.lambda_identity*identity_loss
    if torch.isnan(generator_loss):
            print("⚠️ NaN detected in loss! 2")
    
    return generator_loss

def get_generator_loss_val(opt, model, fake_B, real_B, real_A, loss_plot):

    if opt.lambda_l1_:
        if model.out_channels == 2:
            l1_loss = model.compute_l1_loss(fake_B[:,[0]], real_B)
        else:
            l1_loss = model.compute_l1_loss(fake_B, real_B)
    if opt.lambda_NCE:
        if model.out_channels == 2:
            nce_loss = model.compute_PatchNCELoss(real_A.requires_grad_(requires_grad=False), fake_B[:,[0]])
        else:
            nce_loss = model.compute_PatchNCELoss(real_A.requires_grad_(requires_grad=False), fake_B)
        loss_plot
    if opt.lambda_cycle:
        cycle_loss = model.compute_l1_loss(fake_B, real_B)
        loss_plot['Loss_Val']['cycle'][-1].append((cycle_loss).item())
    if opt.lambda_NGF:
        if model.out_channels == 2:
            NGF_loss = model.compute_NGF_loss(fake_B[:,[0]], real_A, opt.alpha_NGF, NGF = opt.structural_loss, alpha_pixel_wise = fake_B[:,[1]])
        else:
            NGF_loss = model.compute_NGF_loss(fake_B, real_A, opt.alpha_NGF, NGF = opt.structural_loss)
        loss_plot['Loss_Val']['NGF'][-1].append((NGF_loss).item())
    if opt.lambda_perceptual:
        if 'contrary' in opt.perceptual_loss:
            if model.out_channels == 2:
                perceptual_loss = model.compute_perceptual_loss(fake_B[:,[0]], real_B, opt.perceptual_loss)
            else:
                perceptual_loss = model.compute_perceptual_loss(fake_B, real_B, opt.perceptual_loss)
        else:
            if model.out_channels == 2:
                perceptual_loss = model.compute_perceptual_loss(fake_B[:,[0]], real_A, opt.perceptual_loss)
            else:
                perceptual_loss = model.compute_perceptual_loss(fake_B, real_A, opt.perceptual_loss)
        loss_plot['Loss_Val']['Perceptual'][-1].append((opt.lambda_perceptual*perceptual_loss).item())
    if opt.lambda_identity:
        identity_loss = model.compute_identity_loss(real_B)
        structural_loss += opt.lambda_identity*identity_loss
    if model.out_channels == 2:
        discriminator_loss = model.compute_discriminator_loss(real_B, fake_B[:,[0]])
    else:
        discriminator_loss = model.compute_discriminator_loss(real_B, fake_B)
    loss_plot['Loss_Val']['Discriminator'][-1].append((discriminator_loss).item())


def get_generator_loss_cycle(opt, model, fake_B, real_B, fake_A, real_A, cyc_B, cyc_A, pred_fake_B, pred_fake_A):
    generator_loss = 0.0
    cyc_loss_B, identity_loss_B, adv_loss_B, cyc_loss_A, identity_loss_A, adv_loss_A =  model.compute_generator_loss(fake_B, real_B, fake_A, real_A, cyc_B, cyc_A)
    if opt.lambda_l1_:
        generator_loss += opt.lambda_l1_*cyc_loss_B + opt.lambda_l1_*cyc_loss_A
    if opt.lambda_NGF:
        NGF_loss = model.compute_NGF_loss(fake_B, real_A, opt.alpha_NGF)
        generator_loss += opt.lambda_NGF*NGF_loss
    if opt.lambda_gan:
        generator_loss += opt.lambda_gan*(adv_loss_B + adv_loss_A)
    if opt.lambda_identity:
        identity_loss = model.compute_identity_loss(real_B)
        generator_loss += opt.lambda_identity*identity_loss
    discriminator_loss = model.compute_discriminator_loss(real_B, fake_B, real_A, fake_A)
    return generator_loss, discriminator_loss


def get_segmenter_loss(model, pred_seg, real_B, loss_plot):
    with torch.no_grad():
        eps = 10e-8
        weight = torch.ones_like(real_B)
        weight_0 = (real_B == 0).sum()
        weight_1 = (real_B == 1).sum()
        weight[real_B == 0] = (weight_1 + eps) / (weight_1 + weight_0)
        weight[real_B == 1] = (weight_0 + eps) / (weight_1 + weight_0)
    segmenter_loss = torch.nn.BCELoss(weight = weight)(pred_seg, real_B)

    loss_plot['Loss_Train']['BCE'].append(segmenter_loss.item())
    with torch.no_grad():
        pred_seg_ = pred_seg.detach() > 0.5
        accuracy = (real_B == pred_seg_).sum() / (real_B == real_B).sum()
    return segmenter_loss, accuracy

def train_loop(model, ema_model, data_loader, epoch, opt, gpu_device, optimizer, optimizer_D, training = True, loss_plot = None):
    # this is autoencoding for the first epoch, this usually helps the network
    if epoch < opt.warming_epochs and opt.generator:
        real_B_is = "SRC"
        opt.lambda_l1_=100.
    else:
    # this is normal unpaired image translation behaviour
        real_B_is = "TGT"
        opt.lambda_l1_=opt.lambda_l1
    data_loop = tqdm(data_loader)
    data_loop.set_description(f"Epoch [{epoch}/{opt.num_epochs}]")
    model.train()
    if opt.generator:
        if not loss_plot.get('losses'):
            loss_plot['losses'] = ['Discriminator']
            loss_plot['Loss_Train']['Discriminator'] = []
            if opt.lambda_NGF:
                loss_plot['losses'].append('NGF')
                loss_plot['Loss_Train']['NGF'] = []
            if opt.lambda_cycle:
                loss_plot['losses'].append('cycle')
                loss_plot['Loss_Train']['cycle'] = []
            if opt.lambda_gan:
                loss_plot['losses'].append('Gan')
                loss_plot['Loss_Train']['Gan'] = []
                
            if opt.lambda_perceptual:
                loss_plot['losses'].append('Perceptual')
                loss_plot['Loss_Train']['Perceptual'] = []
    else:
        loss_plot['losses'] = ['BCE']
        loss_plot['Loss_Train']['BCE'] = []

    for loss_name in loss_plot['losses']:
        loss_plot['Loss_Train'][loss_name].append([])
            
    
    for batch_idx, real in enumerate(data_loop):
        # Transfer data to the device (CPU or GPU)
        real_A = real["SRC"].to(gpu_device)
        real_B = real[real_B_is].to(gpu_device)
        # changing size of input in case input is 2d
        if opt.spatial_dims==3:
            real_A = real_A.permute((0,1,4,2,3))
            real_B = real_B.permute((0,1,4,2,3))
        else:
            real_A = real_A[:,:,:,:,0]
            real_B = real_B[:,:,:,:,0]
        # Forward pass
        if opt.generator:
            if 'cycle' == model.model:
                generator_loss = 0.0
                for p in model.discriminator_B.parameters():  
                    p.requires_grad = True  
                for p in model.discriminator_A.parameters():  
                    p.requires_grad = True
                fake_B, pred_fake_B, fake_A,  pred_fake_A, cyc_B, cyc_A = model(real_A, real_B, is_training=training)
                if model.out_channels == 2:
                    discriminator_loss = model.compute_discriminator_loss(real_B, fake_B[:,[0]].detach(), )
                else:
                    discriminator_loss = model.compute_discriminator_loss(real_B, fake_B.detach(), real_A, fake_A.detach())
                loss_plot['Loss_Train']['Discriminator'][-1].append((discriminator_loss).item())
                optimizer_D.zero_grad()
                discriminator_loss.backward()
                optimizer_D.step()
                for p in model.discriminator_B.parameters():  
                    p.requires_grad = False  
                for p in model.discriminator_A.parameters():  
                    p.requires_grad = False
                fake_B, pred_fake_B, fake_A,  pred_fake_A, cyc_B, cyc_A = model(real_A, real_B, is_training=training)

                generator_loss = get_generator_loss(opt, model, fake_B, pred_fake_B, real_B, real_A, loss_plot = loss_plot, pred_fake_A = pred_fake_A, cyc_B = cyc_B, cyc_A = cyc_A)
            else:
                generator_loss = 0.0
                for p in model.discriminator_B.parameters():  
                    p.requires_grad = True  
                fake_B, pred_fake_B = model(real_A, real_B, is_training=training)

                if model.out_channels == 2:
                    discriminator_loss = model.compute_discriminator_loss(real_B, fake_B[:,[0]].detach())
                else:
                    discriminator_loss = model.compute_discriminator_loss(real_B, fake_B.detach())
                loss_plot['Loss_Train']['Discriminator'][-1].append((discriminator_loss).item())
                optimizer_D.zero_grad()
                discriminator_loss.backward()
                optimizer_D.step()
                for p in model.discriminator_B.parameters():  
                    p.requires_grad = False  
                fake_B, pred_fake_B = model(real_A, real_B, is_training=training)

                generator_loss = get_generator_loss(opt, model, fake_B, pred_fake_B, real_B, real_A, loss_plot = loss_plot)
                if epoch >= opt.retraining_epoch:
                    fake_B_re, identity_B_re, pred_fake_B_re = model(fake_B, real_B, is_training=training)
                    generator_loss_re, discriminator_loss_re = get_generator_loss(opt, model, fake_B_re, pred_fake_B_re, real_B, real_A)
                    generator_loss += generator_loss_re * opt.lambda_identity_ouput
                    discriminator_loss += discriminator_loss_re * opt.lambda_identity_ouput
                    generator_loss += model.compute_l1_loss(fake_B, fake_B_re) * opt.lambda_identity_ouput
                    generator_loss += model.compute_l1_loss(fake_B, fake_B_re) * opt.lambda_identity_ouput
        else:
            pred_seg = model(real_A)
            segmenter_loss, _ = get_segmenter_loss(model, pred_seg, real_B, loss_plot)

        # Backpropagation and optimization
        optimizer.zero_grad()
        if opt.generator:
            generator_loss.backward()
        else:
            segmenter_loss.backward()
        optimizer.step()
    for loss_name in loss_plot['losses']:
        loss_plot['Loss_Train'][loss_name][-1] = np.mean(loss_plot['Loss_Train'][loss_name][-1])
    if epoch > 100:
        ema_model.update_parameters(model)

def val_loop(model, data_loader, epoch, opt, gpu_device, weights_dir, MSE_self = None, loss_plot = None):
    
    len_val = len(data_loader)
    data_loop = tqdm(data_loader)
    data_loop.set_description(f"Epoch [{epoch}/{opt.num_epochs}]")
    model.eval()
    current_metric = []
    flag_mask_NGF = False
    if opt.generator:
        MSE_self.append([])
        for loss_name in loss_plot['losses']:
            if loss_name == 'Gan':
                continue
            if not loss_plot['Loss_Val'].get(loss_name):
                loss_plot['Loss_Val'][loss_name] = []
            loss_plot['Loss_Val'][loss_name].append([])
    else:
        for loss_name in loss_plot['losses']:
            if not loss_plot['Loss_Val'].get(loss_name):
                loss_plot['Loss_Val'][loss_name] = []
            loss_plot['Loss_Val'][loss_name].append([])
    for batch_idx, real in enumerate(data_loop):
        # Transfer data to the device (CPU or GPU)
        real_A = real["SRC"].to(gpu_device)
        real_B = real['TGT'].to(gpu_device)
        # changing size of input in case input is 2d
        if opt.spatial_dims==3:
            real_A = real_A.permute((0,1,4,2,3))
            real_B = real_B.permute((0,1,4,2,3))
        else:
            real_A = real_A[:,:,:,:,0]
            real_B = real_B[:,:,:,:,0]
        # Forward pass
        if opt.generator:
            fake_B = model(real_A, real_B)
            get_generator_loss_val(opt, model, fake_B, real_B, real_A, loss_plot)
            if fake_B.shape[1] != 1:
                flag_mask_NGF = True
                masks_NGF = torch.nn.Sigmoid()(fake_B[:,[1]])
                masks_NGF = np.uint8(255*masks_NGF[0,0,:,:,].cpu().detach().numpy().squeeze())
                if opt.spatial_dims == 3:
                    masks_NGF = masks_NGF[0]
                fake_B = fake_B[:,[0]]
            fake_B_val_0255 = np.uint8(np.clip(255*(fake_B[0,0,:,:,].cpu().detach().numpy().squeeze()+1)/2, 0,255))
            real_B_val_0255 = np.uint8(np.clip(255*(real_B[0,0,:,:,].cpu().detach().numpy().squeeze()+1)/2, 0,255))
            real_A_val_0255 = np.uint8(np.clip(255*(real_A[0,0,:,:,].cpu().detach().numpy().squeeze()+1)/2, 0,255))
            real_B_ = real_B[0,0].detach().cpu().numpy()
            fake_B_ = fake_B[0,0].detach().cpu().numpy()
            if opt.HM:
                current_psnr = psnr(match_histograms(fake_B_[real_B_!= -1], real_B_[real_B_!= -1]), real_B_[real_B_!= -1], data_range = 2)
                current_metric.append(current_psnr)
            else:
                current_psnr = psnr(fake_B_, real_B_, data_range = 2)
                current_metric.append(current_psnr)
            #print('%.2f dB' % current_psnr)

            if (epoch % 1) == 0:
                if opt.spatial_dims == 3:
                    fake_B_val_0255 = fake_B_val_0255[0]
                    real_B_val_0255 = real_B_val_0255[0]
                    real_A_val_0255 = real_A_val_0255[0]
                fprefix = Path(real['SRC_meta_dict']['filename_or_obj'][0]).name
                if True:
                    if epoch > 21 and (len_val < 20 or batch_idx %  opt.NUM_SAMPLES_MAX == 0):
                        for epoch_before in range(epoch - 20, epoch):
                            fname_img_epoch = weights_dir  / (fprefix+'_fake_B_e%.3d.png' % epoch_before)
                            img_epoch = imageio.imread(fname_img_epoch)
                            MSE_self[-1].append(np.sqrt(np.mean((img_epoch - fake_B_val_0255)**2)))
                    if flag_mask_NGF and (len_val < 20 or batch_idx % opt.NUM_SAMPLES_MAX == 0):
                        fname_out = weights_dir  / (fprefix+'_fake_B_mask_e%.3d.png' % epoch)
                        imageio.imwrite(fname_out, masks_NGF)
                    if len_val < 20 or batch_idx % opt.NUM_SAMPLES_MAX == 0:
                        fname_out = weights_dir  / (fprefix+'_fake_B_e%.3d.png' % epoch)
                        imageio.imwrite(fname_out, fake_B_val_0255)
                        fname_out = weights_dir / (fprefix+'_real_B.png')
                        imageio.imwrite(fname_out, real_B_val_0255)
                        fname_out = weights_dir / (fprefix+'_real_A.png')
                        imageio.imwrite(fname_out, real_A_val_0255)

        else:
            pred_seg = model(real_A)
            segmenter_loss, accuracy = get_segmenter_loss(model, pred_seg, real_B, loss_plot)
            current_metric.append(accuracy.cpu().numpy())
    if opt.generator:
        MSE_self[-1] = np.mean(MSE_self[-1])
        for loss_name in loss_plot['Loss_Val']:
            loss_plot['Loss_Val'][loss_name][-1] = np.mean(loss_plot['Loss_Val'][loss_name][-1])
        plt.figure(0)
        plt.plot(np.arange(epoch-len(MSE_self), epoch), MSE_self)
        plt.savefig(weights_dir.parents[0]  / f'{weights_dir.stem}_MSE.png' )
        plt.close(0)
        plt.figure(1)
        for loss_name in loss_plot['Loss_Val'].keys():
            plt.plot(np.arange(epoch-len(loss_plot['Loss_Val'][loss_name]), epoch), np.array(loss_plot['Loss_Val'][loss_name]), label=loss_name)
        plt.legend()
        plt.savefig(weights_dir.parents[0]  / f'{weights_dir.stem}_Loss_Val.png' )
        plt.close(1)
        plt.figure(2)
        for loss_name in loss_plot['Loss_Train'].keys():
            plt.plot(np.arange(epoch-len(loss_plot['Loss_Train'][loss_name]), epoch), np.array(loss_plot['Loss_Train'][loss_name]), label=loss_name)
        plt.legend()
        plt.savefig(weights_dir.parents[0]  / f'{weights_dir.stem}_Loss_Train.png' )
        plt.close(2)
        plt.close('all')
    else:
        pass
        #for loss_name in loss_plot['Loss_Val'].keys():
        #    plt.plot(np.arange(epoch-len(loss_plot['Loss_Val'][loss_name]), epoch), np.array(loss_plot['Loss_Val'][loss_name]), label=loss_name)
        #plt.legend()
        #plt.savefig(weights_dir.parents[0]  / f'{weights_dir.stem}_Loss_Val.png' )
        #plt.close(1)
        #plt.figure(2)
        #for loss_name in loss_plot['Loss_Train'].keys():
        #    plt.plot(np.arange(epoch-len(loss_plot['Loss_Train'][loss_name]), epoch), np.array(loss_plot['Loss_Train'][loss_name]), label=loss_name)
        #plt.legend()
        #plt.savefig(weights_dir.parents[0]  / f'{weights_dir.stem}_Loss_Train.png' )
        #plt.close(2)
        #plt.close('all')

    return current_metric



def metrics_segmentation(ground_truth, prediction, dict_out):
    if not dict_out.get('Precision'):
        for metric in ['Precision', 'Specificity', 'Sensitivity', 'DICE']:#, 'clDICE']:
            dict_out[metric] = []
    #dict_out['clDICE'].append(clDice(prediction, ground_truth))
    dict_out['Precision'].append(precision_calculation(prediction, ground_truth))
    dict_out['Specificity'].append(specificity_calculation(prediction, ground_truth))
    dict_out['Sensitivity'].append(sensitivity_calculation(prediction, ground_truth))
    dict_out['DICE'].append(dice_coefficient_score_calculation(prediction, ground_truth))


def metrics_generator(ground_truth, prediction, dict_out, label_GT, mask = True):
    if not dict_out.get('PSNR'):
        dict_out['PSNR'] = []
        dict_out['SSIM'] = []
        dict_out['HaarPsi'] = []
        dict_out['MAE'] = []
        if len(ground_truth.shape) == 3:
            dict_out['MS_SSIM'] = []
            dict_out['PSNR_T'] = []
            dict_out['SSIM_T'] = []
            dict_out['HaarPsi_T'] = []
            dict_out['MAE_T'] = []
    if mask:
        mask_GT = ground_truth != -1
    else:
        mask_GT = ground_truth > -100000
    dict_out['PSNR'].append(psnr(ground_truth[mask_GT] , prediction[mask_GT]))
    dict_out['MAE'].append(np.mean(np.abs(ground_truth[mask_GT] - prediction[mask_GT]))*1000)
    if len(ground_truth.shape) == 3:
        vals = np.where(mask_GT != 0)
        bbox = [
            [vals[0].min(), vals[0].max()],
            [vals[1].min(), vals[1].max()],
            [vals[2].min(), vals[2].max()],
        ]
        ground_truth[~mask_GT] = -1
        prediction[~mask_GT] = -1
        prediction_grounded = prediction[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
        ]
        ground_truth_grounded = ground_truth[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
        ]
    else:
        vals = np.where(mask_GT != 0)
        bbox = [
            [vals[0].min(), vals[0].max()],
            [vals[1].min(), vals[1].max()],
        ]
        ground_truth[~mask_GT] = -1
        prediction[~mask_GT] = -1
        prediction_grounded = prediction[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1]
        ]
        ground_truth_grounded = ground_truth[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1]
        ]


    prediction_grounded = (prediction_grounded+1)/2*255
    ground_truth_grounded = (ground_truth_grounded+1)/2*255
    dict_out['SSIM'].append(ssim(prediction_grounded, ground_truth_grounded, data_range = 255))
    
    curr_haar = []
    if len(ground_truth.shape) == 3:
        for i in range(len(ground_truth_grounded)):
            curr_haar.append(haar_psi(ground_truth_grounded[i], prediction_grounded[i])[0])
    else:
        curr_haar.append(haar_psi(ground_truth_grounded, prediction_grounded)[0])
    dict_out['HaarPsi'].append(np.mean(curr_haar))
    if len(ground_truth.shape) == 3:
        print(prediction_grounded.shape)
        print(ground_truth_grounded.shape)
        try:
            dict_out['MS_SSIM'].append(ms_ssim(torch.tensor(prediction_grounded[np.newaxis]),
                        torch.tensor(ground_truth_grounded[np.newaxis])).numpy())
        except:
            dict_out['MS_SSIM'].append(0)

    if len(ground_truth.shape) == 3:
        mask_GT = label_GT != 0
        dict_out['PSNR_T'].append(psnr(ground_truth[mask_GT] , prediction[mask_GT]))
        vals = np.where(mask_GT != 0)
        bbox = [
            [vals[0].min(), vals[0].max()],
            [vals[1].min(), vals[1].max()],
            [vals[2].min(), vals[2].max()],
        ]
        ground_truth[~mask_GT] = -1
        prediction[~mask_GT] = -1
        prediction_grounded = prediction[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
        ]
        ground_truth_grounded = ground_truth[
            bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
        ]

        prediction_grounded = (prediction_grounded+1)/2*255
        ground_truth_grounded = (ground_truth_grounded+1)/2*255

        dict_out['SSIM_T'].append(ssim(prediction_grounded, ground_truth_grounded, data_range = 255))
        curr_haar = []
        for i in range(len(ground_truth_grounded)):
            curr_haar.append(haar_psi(ground_truth_grounded[i], prediction_grounded[i])[0])
        dict_out['HaarPsi_T'].append(np.mean(curr_haar))
        dict_out['MAE_T'].append(np.mean(np.abs(prediction_grounded - ground_truth_grounded)))


