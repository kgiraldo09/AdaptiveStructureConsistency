import torch

def load_model(model, ema_model, path_to_load, start_epoch=0, test=False, lr_scheduler=None, optimizer=None, lr_scheduler_D=None, optimizer_D=None, MSE_self=None, loss_plot = None, best_psnr = 0):
    checkpoint = torch.load(path_to_load)
    model.load_state_dict(checkpoint["model"])
    if checkpoint.get("ema_model"):
        ema_model.load_state_dict(checkpoint["ema_model"])
    if checkpoint.get("epoch"):
            start_epoch = checkpoint["epoch"]
    if not test:
        if checkpoint.get("lr_scheduler"):
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("lr_scheduler_D"):
            lr_scheduler_D.load_state_dict(checkpoint["lr_scheduler_D"])
        if checkpoint.get("optimizer_D"):
            optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        if checkpoint.get("MSE"):
            MSE_self += checkpoint["MSE"]
        if checkpoint.get("loss_plot"):
            loss_plot['Loss_Val'] = checkpoint['loss_plot']['Loss_Val']
            loss_plot['Loss_Train'] = checkpoint['loss_plot']['Loss_Train']
            loss_plot['losses'] = checkpoint['loss_plot']['losses']
        if checkpoint.get("best_psnr"):
            best_psnr =  checkpoint['best_psnr']
    return start_epoch, best_psnr



def save_model(model, ema_model, path_to_save, best=False, epoch=0, lr_scheduler=None, optimizer=None, lr_scheduler_D=None, optimizer_D=None, MSE_self=None, loss_plot = None, best_psnr = None):
    if best:
        print('*' * 20, 'Saving Best', '*' * 20)
        save_dict = {'model' : model.state_dict()}
        ref_name = 'best.pt'
    else:
        if model.generator:
            save_dict = {'model' : model.state_dict(),
                        'epoch': epoch,
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'lr_scheduler_D': lr_scheduler_D.state_dict(),
                        'optimizer_D' : optimizer_D.state_dict(),
                        'ema_model' : ema_model.state_dict(),
                        'MSE' : MSE_self,
                        'loss_plot' : loss_plot,
                        'best_psnr' : best_psnr}
        else:
            save_dict = {'model' : model.state_dict(),
                        'epoch': epoch,
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'ema_model' : ema_model.state_dict(),
                        'loss_plot' : loss_plot,
                        'best_psnr' : best_psnr}
        ref_name = 'last.pt'
    torch.save(save_dict, path_to_save/ref_name)


