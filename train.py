import yaml
import os
import torch
import numpy as np
import argparse
import time
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn

from models.mobrecon_ds import MobRecon_DS_additional
from datasets.hanco_ty import HanCo_ETRI_jitter

from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

import wandb

from utils import *


def main(args, log_every = 500):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ('Current cuda device ', torch.cuda.device_count())

    # checkpoint
    ckpt_dir = "/root/data/minje/mobrecon_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dir = os.path.join(ckpt_dir, args.exp)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # tensorboard
    log_dir = "/root/data/minje/mobrecon_ckpt/tensorboard"
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, args.exp)
    os.makedirs(log_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir)
    # print("Tensorboard log dir: {log_dir}")

 
    train_dataset = HanCo_ETRI_jitter(limit=1e2) # 5e6

    len_dataset = int(len(train_dataset) * 0.95) # 300,000 * 0.2 => 60000
    left_len = len(train_dataset) - len_dataset
    train_dataset, test_dataset = random_split(train_dataset, [len_dataset, left_len])
    test_dataset.mode = 'test' # change mode

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=8, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=8, drop_last=False)

    model = MobRecon_DS_additional()
    model.to(device)

    l2_loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                    betas=(0.9, 0.999), amsgrad=False,
                                    eps=0.00000001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.9)

    train_epoch = range(0, args.epoch)
    for epoch in train_epoch:
        iter = 0
        epoch_loss = 0
        epoch_mpjpe = 0

        model.train()

        for i, item_dict in enumerate(tqdm(train_dataloader)):
            steps = len(train_dataloader) * epoch + i
            iter_loss = 0.
            optimizer.zero_grad()
            
            img = item_dict['image'].float().to(device)
            kps2d = item_dict['keypoints2D'].float().to(device) # normalize in [0, 1]
            kps25d = item_dict['keypoints25D'].float().to(device) # normalize in [0, 1]
            prev = item_dict['prev_additional'].float().to(device)
            add_noise = item_dict['add_noise'].bool().to(device) # for flag
            # import pdb; pdb.set_trace()
            # output_dict keys: keypoints & keypoints3D
            output_dict = model(img, prev) # iter_loss = iter_loss.item()

            # loss calculation
            # 2D loss
            loss_2d = l2_loss(output_dict['keypoints'], kps2d)

            # 2.5D loss
            loss_25d = l2_loss(output_dict['keypoints3D'], kps25d)

            # if prev, prev_loss 
            loss_prev = l2_loss(
                output_dict['keypoints3D'][add_noise],
                prev[add_noise]
            )

            # 240928 update
            # loss between keypoints and keypoints3D[:,:,:2] => 2D
            # loss_inner = l2_loss(output_dict['keypoints3D'][...,:2], output_dict['keypoints'])

            iter_loss = loss_2d + loss_25d + loss_prev * 0.5 # + loss_inner * 0.5
            iter_loss.backward()               
            optimizer.step()
            scheduler.step()

            epoch_loss += iter_loss.item()
            mpjpe = torch.sqrt( ((output_dict['keypoints3D'] - kps25d) ** 2).sum(dim=-1)).mean() * 1000.
            epoch_mpjpe += mpjpe

            iter += 1

            # writer.add_scalar('train iter loss', iter_loss.item(), epoch * len(train_dataloader) + i)
            
            # writer.add_scalar('train iter loss_2d', loss_2d.item(), epoch * len(train_dataloader) + i)
            # writer.add_scalar('train iter loss_25d', loss_25d.item(), epoch * len(train_dataloader) + i)
            # writer.add_scalar('train iter lloss_prevoss', loss_prev.item(), epoch * len(train_dataloader) + i)
            # writer.add_scalar('train iter mpjpe', mpjpe, epoch * len(train_dataloader) + i)

            if steps % log_every == 0:
                log_dict = {
                    'loss': iter_loss.item(),
                    'loss_2d': loss_2d.item(),
                    'loss_25d': loss_25d.item(),
                    'loss_prev': loss_prev.item(),
                    'mpjpe': mpjpe
                }
                
                # Visualize
                gt_25d_keypoints = kps25d.cpu()
                                
                out_2d_keypoints = output_dict['keypoints'].cpu()
                out_25d_keypoints = output_dict['keypoints3D'].cpu()
                
                gt_25d_vis = draw_joint2D(img, gt_25d_keypoints, idx=0)
                                
                pred_2d_vis = draw_joint2D(img, out_2d_keypoints, idx=0)
                pred_25d_vis = draw_joint2D(img, out_25d_keypoints, idx=0)
                
                log_dict['gt'] =   wandb.Image(to_pil_image(gt_25d_vis))
                log_dict['pred_2d'] =  wandb.Image(to_pil_image(pred_2d_vis))
                log_dict['pred_25d'] = wandb.Image(to_pil_image(pred_25d_vis))
                
                wandb.log(log_dict, step=steps)
                
                
        log_dict = {
            'train_loss': epoch_loss / iter,
            'train_mpjpe': epoch_mpjpe / iter
        }           
        wandb.log(log_dict, step=steps)
        # writer.add_scalar('train loss', epoch_loss / iter, epoch )
        # writer.add_scalar('train mpjpe', epoch_mpjpe / iter, epoch )

        if (epoch+1) % 2 == 0:
            save_dir = os.path.join(ckpt_dir, f'{epoch+1}.pt')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict()
            }, save_dir)

        test_iter = 0
        test_epoch_loss = 0
        test_epoch_loss2d = 0
        test_epoch_loss25d = 0
        test_epoch_prev = 0
        test_epoch_mpjpe = 0

        model.eval()

        with torch.no_grad():
            for i, item_dict in enumerate(tqdm(test_dataloader)):
                # steps = len(test_dataloader) * epoch + i
                iter_loss = 0.
                
                img = item_dict['image'].float().to(device)
                kps2d = item_dict['keypoints2D'].float().to(device) # normalize in [0, 1]
                kps25d = item_dict['keypoints25D'].float().to(device) # normalize in [0, 1]
                prev = item_dict['prev_additional'].float().to(device)
                add_noise = item_dict['add_noise'].bool().to(device) # for flag

                output_dict = model(img, prev) # iter_loss = iter_loss.item()

                loss_2d = l2_loss(output_dict['keypoints'], kps2d)

                loss_25d = l2_loss(output_dict['keypoints3D'], kps25d)

                loss_prev = l2_loss(
                    output_dict['keypoints3D'][add_noise],
                    prev[add_noise]
                )

                iter_loss = loss_2d + loss_25d + loss_prev

                test_epoch_loss += iter_loss.item()
                test_epoch_loss2d += loss_2d.item()
                test_epoch_loss25d += loss_25d.item()
                test_epoch_prev += loss_prev.item()
                mpjpe = torch.sqrt( ((output_dict['keypoints3D'] - kps25d) ** 2).sum(dim=-1)).mean() * 1000.
                test_epoch_mpjpe += mpjpe

                test_iter += 1                    

        log_dict = {
            'test_loss': test_epoch_loss / test_iter,
            'test_loss2d': test_epoch_loss2d / test_iter,
            'test_loss25d': test_epoch_loss25d / test_iter,
            'test_loss_prev': test_epoch_prev / test_iter,
            'test_mpjpe': test_epoch_mpjpe / test_iter
        }           
        gt_25d_keypoints = kps25d.cpu()
                        
        out_2d_keypoints = output_dict['keypoints'].cpu()
        out_25d_keypoints = output_dict['keypoints3D'].cpu()
        
        gt_25d_vis = draw_joint2D(img, gt_25d_keypoints, idx=0)
                        
        pred_2d_vis = draw_joint2D(img, out_2d_keypoints, idx=0)
        pred_25d_vis = draw_joint2D(img, out_25d_keypoints, idx=0)
        
        log_dict['test_gt'] =   wandb.Image(to_pil_image(gt_25d_vis))
        log_dict['test_pred_2d'] =  wandb.Image(to_pil_image(pred_2d_vis))
        log_dict['test_pred_25d'] = wandb.Image(to_pil_image(pred_25d_vis))
        
        wandb.log(log_dict, step=steps)

    # writer.close()     


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help='Save experiement name')
    parser.add_argument("--batch", default=50, type=int, dest='batch_size')
    parser.add_argument("--epoch", default=250, type=int, dest='epoch')

    args = parser.parse_args()


    run = wandb.init(project='mobrecon', name=args.exp, job_type='train')
    wandb.run.log_code(root='.',
        include_fn=lambda p: any(p.endswith(ext) for ext in ('.py', '.json', '.yaml', '.md', '.txt.', '.gin')),
        exclude_fn=lambda p: any(s in p for s in ('output', 'tmp', 'wandb', '.git', '.vscode'))
    )
 

    main(args) # throw exp name

    wandb.finish()
    time.sleep(3)