"""
Main training function for single-GPU machines
Default hyper-parameters
+---------+--------------------------------+----------+-------------+---------------+---------+------------+------------+----------+---------------------+
| dataset |        model (dec_bone)        | n_kp_enc | n_kp_prior  | rec_loss_func | beta_kl | kl_balance | patch_size | anchor_s | learned_feature_dim |
+---------+--------------------------------+----------+-------------+---------------+---------+------------+------------+----------+---------------------+
| celeb   | masked (gauss_pointnetpp_feat) |       30 |          50 | vgg           |      40 |      0.001 |          8 |    0.125 |                  10 |
| traffic | object (gauss_pointnetpp)      |       15 |          20 | vgg           |      30 |      0.001 |         16 |     0.25 |                  20 |
| clevrer | object (gauss_pointnetpp)      |       10 |          20 | vgg           |      40 |      0.001 |         16 |     0.25 |                   5 |
| shapes  | object (gauss_pointnetpp)      |        8 |          15 | mse           |    0.1 |      0.001 |          8 |     0.25 |                   5 |
+---------+--------------------------------+----------+-------------+---------------+---------+------------+------------+----------+---------------------+
"""
# imports
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import argparse
# torch
import torch
import torch.nn.functional as F
from utils.loss_functions import ChamferLossKL, calc_kl, calc_reconstruction_loss, VGGDistance
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim
# modules
from models import KeyPointVAE
# datasets
from dataset.celeba_dataset import CelebAPrunedAligned_MAFLVal, evaluate_lin_reg_on_mafl
from dataset.traffic_ds import TrafficDataset
from dataset.clevrer_ds import CLEVRERDataset
from dataset.shapes_ds import generate_shape_dataset_torch
# util functions
from utils.util_func import plot_keypoints_on_image_batch, create_masks_fast, prepare_logdir, save_config, log_line,\
    plot_bb_on_image_batch_from_masks_nms
from eval.eval_model import evaluate_validation_elbo

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from utils_loss import calculate_reconstruction_loss,evaluate,evaluate_sup_linear_reg,plot


def train_dlp(ds="shapes", batch_size=16, lr=5e-4, device=torch.device("cpu"), kp_activation="none",
              pad_mode='replicate', num_epochs=250, load_model=False, n_kp=8, recon_loss_type="mse",
              use_logsoftmax=False, sigma=0.1, beta_kl=1.0, beta_rec=1.0, dropout=0.0,
              dec_bone="gauss_pointnetpp", patch_size=16, topk=15, n_kp_enc=20, eval_epoch_freq=5,
              learned_feature_dim=0, n_kp_prior=100, weight_decay=0.0, kp_range=(0, 1),
              run_prefix="", mask_threshold=0.2, use_tps=False, use_pairs=False, use_object_enc=True,
              use_object_dec=False, warmup_epoch=5, iou_thresh=0.2, anchor_s=0.25, learn_order=False,
              kl_balance=0.1, exclusive_patches=False):
    """
    ds: dataset name (str)
    enc_channels: channels for the posterior CNN (takes in the whole image)
    prior_channels: channels for prior CNN (takes in patches)
    n_kp: number of kp to extract from each (!) patch
    n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
    n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
    use_logsoftmax: for spatial-softmax, set True to use log-softmax for numerical stability
    pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
    sigma: the prior std of the KP
    dropout: dropout for the CNNs. We don't use it though...
    dec_bone: decoder backbone -- "gauss_pointnetpp_feat": Masked Model, "gauss_pointnetpp": Object Model
    patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
    kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
    learned_feature_dim: the latent visual features dimensions extracted from glimpses.
    kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
    mask_threshold: activation threshold (>thresh -> 1, else 0) for the binary mask created from the Gaussian-maps.
    anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
    learn_order: experimental feature to learn the order of keypoints - but it doesn't work yet.
    use_object_enc: set True to use a separate encoder to encode visual features of glimpses.
    use_object_dec: set True to use a separate decoder to decode glimpses (Object Model).
    iou_thresh: intersection-over-union threshold for non-maximal suppression (nms) to filter bounding boxes
    use_tps: set True to use a tps augmentation on the input image for datasets that support this option
    use_pairs: for CelebA dataset, set True to use a tps-augmented image for the prior.
    topk: the number top-k particles with the lowest variance (highest confidence) to filter for the plots.
    warmup_epoch: (used for the Object Model) number of epochs where only the object decoder is trained.
    recon_loss_type: tpe of pixel reconstruction loss ("mse", "vgg").
    beta_rec: coefficient for the reconstruction loss (we use 1.0).
    beta_kl: coefficient for the KL divergence term in the loss.
    kl_balance: coefficient for the balance between the ChamferKL (for the KP)
                and the standard KL (for the visual features),
                kl_loss = beta_kl * (chamfer_kl + kl_balance * kl_features)
    exclusive_patches: (mostly) enforce one particle pre object by masking up regions that were already encoded.
    """

    # load data
    image_size = 128
    imwidth = 160
    crop = 16
    ch = 3
    enc_channels = [32, 64, 128, 256]
    prior_channels = (16, 32, 64)
    root = '/home/kiki/dados_servidor/Servidor/Art/landmarks/celeba'
    if use_tps:
        import utils.tps as tps
        if use_pairs:
            warper = tps.Warper(H=imwidth, W=imwidth, im1_multiplier=0.1, im1_multiplier_aff=0.1)
        else:
            warper = tps.WarperSingle(H=imwidth, W=imwidth, warpsd_all=0.001, warpsd_subset=0.01, transsd=0.1,
                                        scalesd=0.1, rotsd=5)
        print('using tps augmentation')
    else:
        warper = None
    dataset = CelebAPrunedAligned_MAFLVal(root=root, train=True, do_augmentations=False, imwidth=imwidth, crop=crop,
                                            pair_warper=warper)
    milestones = (50, 100, 200)


    # save hyper-parameters
    hparams = {'ds': ds, 'batch_size': batch_size, 'lr': lr, 'kp_activation': kp_activation, 'pad_mode': pad_mode,
               'num_epochs': num_epochs, 'n_kp': n_kp, 'recon_loss_type': recon_loss_type,
               'use_logsoftmax': use_logsoftmax, 'sigma': sigma, 'beta_kl': beta_kl, 'beta_rec': beta_rec,
               'dec_bone': dec_bone, 'patch_size': patch_size, 'topk': topk, 'n_kp_enc': n_kp_enc,
               'eval_epoch_freq': eval_epoch_freq, 'learned_feature_dim': learned_feature_dim,
               'n_kp_prior': n_kp_prior, 'weight_decay': weight_decay, 'kp_range': kp_range,
               'run_prefix': run_prefix, 'mask_threshold': mask_threshold, 'use_tps': use_tps, 'use_pairs': use_pairs,
               'use_object_enc': use_object_enc, 'use_object_dec': use_object_dec, 'warmup_epoch': warmup_epoch,
               'iou_thresh': iou_thresh, 'anchor_s': anchor_s, 'learn_order': learn_order, 'kl_balance': kl_balance,
               'milestones': milestones, 'image_size': image_size, 'enc_channels': enc_channels,
               'prior_channels': prior_channels, 'exclusive_patches': exclusive_patches}

    # create dataloader
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True,
                            drop_last=True)
    # model
    model = KeyPointVAE(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                        image_size=image_size, n_kp=n_kp, learned_feature_dim=learned_feature_dim,
                        use_logsoftmax=use_logsoftmax, pad_mode=pad_mode, sigma=sigma,
                        dropout=dropout, dec_bone=dec_bone, patch_size=patch_size, n_kp_enc=n_kp_enc,
                        n_kp_prior=n_kp_prior, kp_range=kp_range, kp_activation=kp_activation,
                        mask_threshold=mask_threshold, use_object_enc=use_object_enc,
                        exclusive_patches=exclusive_patches, use_object_dec=use_object_dec, anchor_s=anchor_s,
                        learn_order=learn_order).to(device)

    logvar_p = torch.log(torch.tensor(sigma ** 2)).to(device)  # logvar of the constant std -> for the kl
    # prepare saving location
    run_name = f'{ds}_dlp_{dec_bone}' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')
    save_config(log_dir, hparams)

    kl_loss_func = ChamferLossKL(use_reverse_kl=False)
    if recon_loss_type == "vgg":
        recon_loss_func = VGGDistance(device=device)
    else:
        recon_loss_func = calc_reconstruction_loss
    betas = (0.9, 0.999)
    eps = 1e-4
    # we use separate optimizers for the encoder and decoder, but it is not really necessary...
    optimizer_e = optim.Adam(model.get_parameters(encoder=True, prior=True, decoder=False), lr=lr, betas=betas, eps=eps,
                             weight_decay=weight_decay)
    optimizer_d = optim.Adam(model.get_parameters(encoder=False, prior=False, decoder=True), lr=lr, betas=betas,
                             eps=eps, weight_decay=weight_decay)

    scheduler_e = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=milestones, gamma=0.5)
    scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=milestones, gamma=0.5)

    if load_model:
        try:
            model.load_state_dict(
                torch.load(os.path.join(save_dir, f'{ds}_dlp_{dec_bone}.pth'), map_location=device))
            print("loaded model from checkpoint")
        except:
            print("model checkpoint not found")

    # statistics
    losses = []
    losses_rec = []
    losses_kl = []
    losses_kl_kp = []
    losses_kl_feat = []

    # initialize linear regression statistics (celeba)
    linreg_error = best_linreg_error = 1.0
    best_linreg_epoch = 0
    linreg_logvar_error = best_linreg_logvar_error = 1.0
    best_linreg_logvar_epoch = 0
    linreg_features_error = best_linreg_features_error = 1.0
    best_linreg_features_epoch = 0

    linreg_errors = []
    linreg_logvar_errors = []
    linreg_features_errors = []

    # initialize validation statistics
    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0

    # save PSNR values of the reconstruction
    psnrs = []

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        batch_losses_rec = []
        batch_losses_kl = []
        batch_losses_kl_kp = []
        batch_losses_kl_feat = []
        batch_psnrs = []
        psnr=0
        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            if len(batch['data'].shape) == 5:
                x_prior = batch['data'][:, 0].to(device)
                x = batch['data'][:, 1].to(device)
            else:
                x = batch['data'].to(device)
                x_prior = x
        
            batch_size = x.shape[0]
            # forward pass
            noisy_masks = (epoch < 5 * warmup_epoch)  # add small noise to the alpha masks
            # model_output = model(x, x_prior=x_prior, warmup=(epoch < warmup_epoch), noisy_masks=noisy_masks)
            model_output = model(x, x_prior=x_prior, warmup=False, noisy_masks=noisy_masks)
            mu_p = model_output['kp_p']
            gmap = model_output['gmap']
            mu = model_output['mu']
            logvar = model_output['logvar']
            rec_x = model_output['rec']
            mu_features = model_output['mu_features']
            logvar_features = model_output['logvar_features']
            # object stuff
            dec_objects_original = model_output['dec_objects_original']
            cropped_objects_original = model_output['cropped_objects_original']
            obj_on = model_output['obj_on']  # [batch_size, n_kp]

            loss_rec,batch_psnrs,psnr= calculate_reconstruction_loss(epoch,warmup_epoch,
                                            x, rec_x,recon_loss_func,recon_loss_type,batch_psnrs,
                                            psnr, use_object_dec, dec_objects_original, 
                                            cropped_objects_original)


            # kl-divergence
            logvar_kp = logvar_p.expand_as(mu_p)

            # the final kp is the bg kp which is located in the center (so no need for it)
            # to reproduce the results on celeba, use `mu_post = mu`, `logvar_post = logvar`
            mu_post = mu[:, :-1]
            logvar_post = logvar[:, :-1]
            # mu_post = mu
            # logvar_post = logvar
            mu_prior = mu_p
            logvar_prior = logvar_kp

            loss_kl_kp = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
                                      logvar_gts=logvar_prior).mean()

            if learned_feature_dim > 0:
                loss_kl_feat = calc_kl(logvar_features.view(-1, logvar_features.shape[-1]),
                                       mu_features.view(-1, mu_features.shape[-1]), reduce='none')
                loss_kl_feat = loss_kl_feat.view(batch_size, n_kp_enc + 1).sum(1).mean()
            else:
                loss_kl_feat = torch.tensor(0.0, device=device)
            loss_kl = loss_kl_kp + kl_balance * loss_kl_feat

            loss = beta_rec * loss_rec + beta_kl * loss_kl
            # backprop
            optimizer_e.zero_grad()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_e.step()
            optimizer_d.step()
            # log
            batch_losses.append(loss.data.cpu().item())
            batch_losses_rec.append(loss_rec.data.cpu().item())
            batch_losses_kl.append(loss_kl.data.cpu().item())
            batch_losses_kl_kp.append(loss_kl_kp.data.cpu().item())
            batch_losses_kl_feat.append(loss_kl_feat.data.cpu().item())
            # progress bar
            if use_object_dec and epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            elif use_object_dec and noisy_masks:
                pbar.set_description_str(f'epoch #{epoch} (noisy masks)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')
            pbar.set_postfix(loss=loss.data.cpu().item(), rec=loss_rec.data.cpu().item(),
                             kl=loss_kl.data.cpu().item())
        pbar.close()
        losses.append(np.mean(batch_losses))
        losses_rec.append(np.mean(batch_losses_rec))
        losses_kl.append(np.mean(batch_losses_kl))
        losses_kl_kp.append(np.mean(batch_losses_kl_kp))
        losses_kl_feat.append(np.mean(batch_losses_kl_feat))
        if len(batch_psnrs) > 0:
            psnrs.append(np.mean(batch_psnrs))
        # keep track of bounding box scores to set a hard threshold (as bb scores are not normalized)
        # epoch_bb_scores = torch.cat(batch_bb_scores, dim=0)
        # bb_mean_score = epoch_bb_scores.mean().data.cpu().item()
        # bb_mean_scores.append(bb_mean_score)
        # schedulers
        scheduler_e.step()
        scheduler_d.step()
        # epoch summary
        log_str = f'epoch {epoch} summary for dec backbone: {dec_bone}\n'
        log_str += f'loss: {losses[-1]:.3f}, rec: {losses_rec[-1]:.3f}, kl: {losses_kl[-1]:.3f}\n'
        log_str += f'kl_balance: {kl_balance:.4f}, kl_kp: {losses_kl_kp[-1]:.3f}, kl_feat: {losses_kl_feat[-1]:.3f}\n'
        log_str += f'mu max: {mu.max()}, mu min: {mu.min()}\n'
        if ds != 'celeb':
            log_str += f'val loss (freq: {eval_epoch_freq}): {valid_loss:.3f},' \
                       f' best: {best_valid_loss:.3f} @ epoch: {best_valid_epoch}\n'
        if obj_on is not None:
            log_str += f'obj_on max: {obj_on.max()}, obj_on min: {obj_on.min()}\n'
        if len(psnrs) > 0:
            log_str += f'mean psnr: {psnrs[-1]:.3f}\n'
        print(log_str)
        log_line(log_dir, log_str)

        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            evaluate(x,x_prior,model,model_output,mu,mu_p,topk,mask_threshold,iou_thresh,
                     logvar,rec_x,dec_objects_original,cropped_objects_original,use_object_dec,
                     kp_range,epoch,fig_dir,log_dir,save_dir,ds,dec_bone,run_prefix,device)
            linreg_error,linreg_logvar_error,linreg_features_error=evaluate_sup_linear_reg(model,
                                                                 root, device, image_size, fig_dir,
                                                                   epoch,learned_feature_dim=0,save_dir=None,
                                                                   ds=None,dec_bone=None,run_prefix=None,log_dir=None)

        linreg_errors.append(linreg_error * 100)
        linreg_logvar_errors.append(linreg_logvar_error * 100)
        linreg_features_errors.append(linreg_features_error * 100)
        valid_losses.append(valid_loss)
        # plot graphs
        if epoch > 0:
            plot(losses,losses_kl,losses_kl_kp,losses_kl_feat,losses_rec,
                 linreg_errors,linreg_logvar_errors,linreg_features_errors,
                 valid_losses,run_name,fig_dir,learned_feature_dim,ds)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLP Single-GPU Training")
    parser.add_argument("-d", "--dataset", type=str, default='celeb',
                        help="dataset of to train the model on: ['celeb', 'traffic', 'clevrer', 'shapes']")
    parser.add_argument("-o", "--override", action='store_true',
                        help="set True to override default hyper-parameters via command line")
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=2e-4)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=100)
    parser.add_argument("-e", "--eval_freq", type=int, help="evaluation epoch frequency", default=2)
    parser.add_argument("-s", "--sigma", type=float, help="the prior std of the KP", default=0.1)
    parser.add_argument("-p", "--prefix", type=str, help="string prefix for logging", default="")
    parser.add_argument("-r", "--beta_rec", type=float, help="beta coefficient for the reconstruction loss",
                        default=1.0)
    parser.add_argument("-k", "--beta_kl", type=float, help="beta coefficient for the kl divergence",
                        default=1.0)
    parser.add_argument("-c", "--kl_balance", type=float,
                        help="coefficient for the balance between the ChamferKL (for the KP) and the standard KL",
                        default=0.001)
    parser.add_argument("-v", "--rec_loss_function", type=str, help="type of reconstruction loss: 'mse', 'vgg'",
                        default="mse")
    parser.add_argument("--n_kp_enc", type=int, help="number of posterior kp to be learned", default=30)
    parser.add_argument("--n_kp_prior", type=int, help="number of kp to filter from the set of prior kp", default=50)
    parser.add_argument("--dec_bone", type=str,
                        help="decoder backbone:'gauss_pointnetpp_feat': Masked Model, 'gauss_pointnetpp': Object Model",
                        default="gauss_pointnetpp")
    parser.add_argument("--patch_size", type=int,
                        help="patch size for the prior KP proposals network (not to be confused with the glimpse size)",
                        default=8)
    parser.add_argument("--learned_feature_dim", type=int,
                        help="the latent visual features dimensions extracted from glimpses",
                        default=10)
    parser.add_argument("--use_object_enc", action='store_true',
                        help="set True to use a separate encoder to encode visual features of glimpses")
    parser.add_argument("--use_object_dec", action='store_true',
                        help="set True to use a separate decoder to decode glimpses (Object Model)")
    parser.add_argument("--warmup_epoch", type=int,
                        help="number of epochs where only the object decoder is trained",
                        default=2)
    parser.add_argument("--anchor_s", type=float,
                        help="defines the glimpse size as a ratio of image_size", default=0.25)
    parser.add_argument("--exclusive_patches", action='store_true',
                        help="set True to enable non-overlapping object patches")
    args = parser.parse_args()

    # default hyper-parameters
    lr = 2e-4
    batch_size = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    load_model = False
    eval_epoch_freq = 2
    n_kp = 1  # num kp per patch
    mask_threshold = 0.2  # mask threshold for the features from the encoder
    kp_range = (-1, 1)
    weight_decay = 0.0
    run_prefix = ""
    learn_order = False
    use_logsoftmax = False
    pad_mode = 'replicate'
    sigma = 0.1  # default sigma for the gaussian maps
    dropout = 0.0
    kp_activation = "tanh"
    # dataset specific
    ds = args.dataset
    if args.dataset == 'celeb':
        beta_kl = 40.0
        beta_rec = 1.0
        n_kp_enc = 30  # total kp to output from the encoder / filter from prior
        n_kp_prior = 50
        # patch_size = 8
        patch_size = 16 
        learned_feature_dim = 10  # additional features than x,y for each kp
        dec_bone = "gauss_pointnetpp_feat"
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "vgg"
        use_tps = True
        use_pairs = True
        use_object_enc = True  # separate object encoder
        use_object_dec = False  # separate object decoder
        warmup_epoch = 0
        anchor_s = 0.125
        kl_balance = 0.001
        exclusive_patches = False
    elif args.dataset == 'traffic':
        beta_kl = 30.0
        beta_rec = 1.0
        n_kp_enc = 15  # total kp to output from the encoder / filter from prior
        n_kp_prior = 20
        patch_size = 16
        learned_feature_dim = 10  # additional features than x,y for each kp
        dec_bone = "gauss_pointnetpp"
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "vgg"
        use_tps = False
        use_pairs = False
        use_object_enc = True  # separate object encoder
        use_object_dec = True  # separate object decoder
        warmup_epoch = 2
        anchor_s = 0.25
        kl_balance = 0.001
        exclusive_patches = False
    elif args.dataset == 'clevrer':
        beta_kl = 40.0
        beta_rec = 1.0
        n_kp_enc = 10  # total kp to output from the encoder / filter from prior
        n_kp_prior = 20
        patch_size = 16
        learned_feature_dim = 5  # additional features than x,y for each kp
        dec_bone = "gauss_pointnetpp"
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "vgg"
        use_tps = False
        use_pairs = False
        use_object_enc = True  # separate object encoder
        use_object_dec = True  # separate object decoder
        warmup_epoch = 1
        anchor_s = 0.25
        kl_balance = 0.001
        exclusive_patches = False
    elif args.dataset == 'shapes':
        beta_kl = 0.05
        beta_rec = 1.0
        n_kp_enc = 10  # total kp to output from the encoder / filter from prior
        n_kp_prior = 15
        patch_size = 16
        learned_feature_dim = 6  # additional features than x,y for each kp
        dec_bone = "gauss_pointnetpp"
        topk = min(10, n_kp_enc)  # display top-10 kp with smallest variance
        recon_loss_type = "mse"
        use_tps = False
        use_pairs = False
        use_object_enc = True  # separate object encoder
        use_object_dec = True  # separate object decoder
        warmup_epoch = 1
        anchor_s = 0.25
        kl_balance = 0.001
        exclusive_patches = True
        # override manually
        lr = 1e-3
        batch_size = 64
    else:
        raise NotImplementedError("unrecognized dataset, please implement it and add it to the train script")

    override_hp = args.override
    if override_hp:
        lr = args.lr
        batch_size = args.batch_size
        num_epochs = args.num_epochs
        eval_epoch_freq = args.eval_freq
        run_prefix = args.prefix
        sigma = args.sigma
        beta_kl = args.beta_kl
        beta_rec = args.beta_rec
        n_kp_enc = args.n_kp_enc
        n_kp_prior = args.n_kp_prior
        patch_size = args.patch_size
        learned_feature_dim = args.learned_feature_dim
        dec_bone = args.dec_bone
        recon_loss_type = args.rec_loss_function
        use_object_enc = args.use_object_enc
        use_object_dec = args.use_object_dec
        warmup_epoch = args.warmup_epoch
        anchor_s = args.anchor_s
        kl_balance = args.kl_balance
        exclusive_patches = args.exclusive_patches

    model = train_dlp(ds=ds, batch_size=batch_size, lr=lr,
                      device=device, num_epochs=num_epochs, kp_activation=kp_activation,
                      load_model=load_model, n_kp=n_kp, use_logsoftmax=use_logsoftmax, pad_mode=pad_mode,
                      sigma=sigma, beta_kl=beta_kl, beta_rec=beta_rec, dropout=dropout, dec_bone=dec_bone,
                      kp_range=kp_range, learned_feature_dim=learned_feature_dim, weight_decay=weight_decay,
                      recon_loss_type=recon_loss_type, patch_size=patch_size, topk=topk, n_kp_enc=n_kp_enc,
                      eval_epoch_freq=eval_epoch_freq, n_kp_prior=n_kp_prior, run_prefix=run_prefix,
                      mask_threshold=mask_threshold, use_tps=use_tps, use_pairs=use_pairs, anchor_s=anchor_s,
                      use_object_enc=use_object_enc, use_object_dec=use_object_dec, exclusive_patches=exclusive_patches,
                      warmup_epoch=warmup_epoch, learn_order=learn_order, kl_balance=kl_balance)
