import torch
from torch.nn import functional as F
import os
from utils.loss_functions import ChamferLossKL, calc_kl, calc_reconstruction_loss, VGGDistance

# util functions
from utils.util_func import plot_keypoints_on_image_batch, create_masks_fast, prepare_logdir, save_config, log_line,\
    plot_bb_on_image_batch_from_masks_nms
from eval.eval_model import evaluate_validation_elbo
from torchvision import utils as vutils

def calculate_reconstruction_loss(epoch,warmup_epoch,x, rec_x,recon_loss_func,recon_loss_type,batch_psnrs,psnr, use_object_dec, dec_objects_original, cropped_objects_original):
    # reconstruction error
    if use_object_dec and dec_objects_original is not None and epoch < warmup_epoch:
        # reconstruct patches in the warmup stage
        if recon_loss_type == "vgg":
            _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
            dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
            cropped_objects_original = cropped_objects_original.reshape(-1,
                                                                        *cropped_objects_original.shape[2:])
            # vgg has a minimal input size, so we interpolate if the patch is too small
            if cropped_objects_original.shape[-1] < 32:
                cropped_objects_original = F.interpolate(cropped_objects_original, size=32, mode='bilinear',
                                                         align_corners=False)
                dec_objects_rgb = F.interpolate(dec_objects_rgb, size=32, mode='bilinear',
                                                align_corners=False)
            loss_rec_obj = recon_loss_func(cropped_objects_original, dec_objects_rgb, reduction="mean")

        else:
            _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
            dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
            cropped_objects_original = cropped_objects_original.clone().reshape(-1,
                                                                                *cropped_objects_original.shape[
                                                                                 2:])
            loss_rec_obj = calc_reconstruction_loss(cropped_objects_original, dec_objects_rgb,
                                                    loss_type='mse', reduction='mean')

        loss_rec = loss_rec_obj
    else:
        # reconstruct full image
        if recon_loss_type == "vgg":
            loss_rec = recon_loss_func(x, rec_x, reduction="mean")
        else:
            loss_rec = calc_reconstruction_loss(x, rec_x, loss_type='mse', reduction='mean')

        with torch.no_grad():
            psnr = -10 * torch.log10(F.mse_loss(rec_x, x))
            batch_psnrs.append(psnr.data.cpu().item())

    return loss_rec,batch_psnrs,psnr


def evaluate(x,x_prior,model,model_output,mu,mu_p,topk,mask_threshold,iou_thresh,logvar,rec_x,dec_objects_original,cropped_objects_original,use_object_dec,kp_range,epoch,fig_dir,log_dir,save_dir,ds,dec_bone,run_prefix,device):
    max_imgs = 8
    img_with_kp = plot_keypoints_on_image_batch(mu[:, :-1].clamp(min=kp_range[0], max=kp_range[1]),
                                                x, radius=3, thickness=1, max_imgs=max_imgs, kp_range=kp_range)
    img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x_prior, radius=3, thickness=1, max_imgs=max_imgs,
                                                    kp_range=kp_range)
    # top-k
    with torch.no_grad():
        logvar_sum = logvar[:, :-1].sum(-1)
        logvar_topk = torch.topk(logvar_sum, k=topk, dim=-1, largest=False)
        indices = logvar_topk[1]  # [batch_size, topk]
        batch_indices = torch.arange(mu.shape[0]).view(-1, 1).to(mu.device)
        topk_kp = mu[batch_indices, indices]
        # bounding boxes
        masks = create_masks_fast(mu[:, :-1].detach(), anchor_s=model.anchor_s, feature_dim=x.shape[-1])
        masks = torch.where(masks < mask_threshold, 0.0, 1.0)
        bb_scores = -1 * logvar_sum
        hard_threshold = bb_scores.mean()
    if use_object_dec:
        img_with_masks_nms, nms_ind = plot_bb_on_image_batch_from_masks_nms(masks, x, scores=bb_scores,
                                                                            iou_thresh=iou_thresh,
                                                                            thickness=1, max_imgs=max_imgs,
                                                                            hard_thresh=hard_threshold)
        # hard_thresh: a general threshold for bb scores (set None to not use it)
        bb_str = f'bb scores: max: {bb_scores.max():.2f}, min: {bb_scores.min():.2f},' \
                    f' mean: {bb_scores.mean():.2f}\n'
        print(bb_str)
    log_line(log_dir, bb_str)
    img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                        radius=3, thickness=1, max_imgs=max_imgs,
                                                        kp_range=kp_range)
    if use_object_dec and dec_objects_original is not None:
        dec_objects = model_output['dec_objects']
        vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(device),
                                        rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(device),
                                        img_with_kp_topk[:max_imgs, -3:].to(device),
                                        dec_objects[:max_imgs, -3:],
                                        img_with_masks_nms[:max_imgs, -3:].to(device)],
                                    dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
                            nrow=8, pad_value=1)
        with torch.no_grad():
            _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
            dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
            cropped_objects_original = cropped_objects_original.clone().reshape(-1, 3,
                                                                                cropped_objects_original.shape[
                                                                                    -1],
                                                                                cropped_objects_original.shape[
                                                                                    -1])
            if cropped_objects_original.shape[-1] != dec_objects_rgb.shape[-1]:
                cropped_objects_original = F.interpolate(cropped_objects_original,
                                                            size=dec_objects_rgb.shape[-1],
                                                            align_corners=False, mode='bilinear')
        vutils.save_image(
            torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                        dim=0).data.cpu(), '{}/image_obj_{}.jpg'.format(fig_dir, epoch),
            nrow=8, pad_value=1)
    else:
        vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(device),
                                        rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(device),
                                        img_with_kp_topk[:max_imgs, -3:].to(device)],
                                    dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
                            nrow=8, pad_value=1)
    torch.save(model.state_dict(),
                os.path.join(save_dir, f'{ds}_dlp_{dec_bone}{run_prefix}.pth'))

# from utils import evaluate_lin_reg_on_mafl
from dataset.celeba_dataset import CelebAPrunedAligned_MAFLVal, evaluate_lin_reg_on_mafl

def evaluate_sup_linear_reg(model, root, device, image_size, fig_dir, epoch,learned_feature_dim=0,save_dir=None,ds=None,dec_bone=None,run_prefix=None,log_dir=None):
    # evaluate supervised linear regression errors
    eval_model = model
    print("evaluating linear regression error...")
    linreg_error_train, linreg_error = evaluate_lin_reg_on_mafl(eval_model, root=root, use_logvar=False,
                                                                batch_size=100,
                                                                device=device, img_size=image_size,
                                                                fig_dir=fig_dir,
                                                                epoch=epoch)
    if best_linreg_error > linreg_error:
        best_linreg_error = linreg_error
        best_linreg_epoch = epoch
    linreg_logvar_error_train, linreg_logvar_error = evaluate_lin_reg_on_mafl(eval_model, root=root,
                                                                                use_logvar=True,
                                                                                batch_size=100,
                                                                                device=device,
                                                                                img_size=image_size,
                                                                                fig_dir=fig_dir,
                                                                                epoch=epoch)
    if best_linreg_logvar_error > linreg_logvar_error:
        best_linreg_logvar_error = linreg_logvar_error
        best_linreg_logvar_epoch = epoch
    if learned_feature_dim > 0:
        linreg_features_error_train, linreg_features_error = evaluate_lin_reg_on_mafl(eval_model,
                                                                                        root=root,
                                                                                        use_logvar=True,
                                                                                        batch_size=100,
                                                                                        device=device,
                                                                                        img_size=image_size,
                                                                                        fig_dir=fig_dir,
                                                                                        epoch=epoch,
                                                                                        use_features=True)
        if best_linreg_features_error > linreg_features_error:
            best_linreg_features_error = linreg_features_error
            best_linreg_features_epoch = epoch
            torch.save(model.state_dict(),
                        os.path.join(save_dir,
                                    f'{ds}_dlp_{dec_bone}{run_prefix}_best.pth'))
    linreg_str = f'eval epoch {epoch}: error: {linreg_error * 100:.4f}%,' \
                    f' error with logvar: {linreg_logvar_error * 100:.4f},' \
                    f' train logvar error: {linreg_logvar_error_train * 100:.4f}%\n'
    if learned_feature_dim > 0:
        linreg_str += f'error with features: {linreg_features_error * 100:.4f}%,' \
                        f' train logvar error: {linreg_features_error_train * 100:.4f}%\n'
    linreg_str += f'best error {best_linreg_epoch}: {best_linreg_error * 100:.4f}%,' \
                    f'  error with logvar {best_linreg_logvar_epoch}: {best_linreg_logvar_error * 100:.4f}%\n'
    if learned_feature_dim > 0:
        linreg_str += f'error with features' \
                        f' {best_linreg_features_epoch}: {best_linreg_features_error * 100:.4f}%\n'
    print(linreg_str)
    log_line(log_dir, linreg_str)
    return linreg_error, linreg_logvar_error, linreg_features_error

import numpy as np
import matplotlib.pyplot as plt

def plot(losses,losses_kl,losses_kl_kp,losses_kl_feat,losses_rec,linreg_errors,linreg_logvar_errors,linreg_features_errors,valid_losses,run_name,fig_dir,learned_feature_dim,ds):
    # plot graphs
    num_plots = 4
    fig = plt.figure()
    ax = fig.add_subplot(num_plots, 1, 1)
    ax.plot(np.arange(len(losses[1:])), losses[1:], label="loss")
    ax.set_title(run_name)
    ax.legend()

    ax = fig.add_subplot(num_plots, 1, 2)
    ax.plot(np.arange(len(losses_kl[1:])), losses_kl[1:], label="kl", color='red')
    if learned_feature_dim > 0:
        ax.plot(np.arange(len(losses_kl_kp[1:])), losses_kl_kp[1:], label="kl_kp", color='cyan')
        ax.plot(np.arange(len(losses_kl_feat[1:])), losses_kl_feat[1:], label="kl_feat", color='green')
    ax.legend()

    ax = fig.add_subplot(num_plots, 1, 3)
    ax.plot(np.arange(len(losses_rec[1:])), losses_rec[1:], label="rec", color='green')
    ax.legend()

    ax = fig.add_subplot(num_plots, 1, 4)
    ax.plot(np.arange(len(linreg_errors[1:])), linreg_errors[1:], label="linreg_err %")
    ax.plot(np.arange(len(linreg_logvar_errors[1:])), linreg_logvar_errors[1:], label="linreg_v_err %")
    if learned_feature_dim > 0:
        ax.plot(np.arange(len(linreg_features_errors[1:])), linreg_features_errors[1:],
                label="linreg_f_err %")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{run_name}_graph.jpg')
    plt.close('all')