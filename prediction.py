import os
import torch
import random
import argparse

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

from torch.utils import data
from torch.optim import Adam

from helper import get_options, load_yaml, reconstruct_from_patches,reconstruct_from_trim_patches
from utils import L0_Loss
from model_zoo.network_unet3d import UNet3D
from dataset_n2n_3d import Noisy_volume_dataset

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./prediction_config.yaml',
                    help='Path to YAML configuration file.')

args = parser.parse_args()
opt = load_yaml(args.config)
opt_train, opt_model, opt_dataset = get_options(opt)
device = torch.device('cuda' if opt['init']['gpu_ids'] is not None else 'cpu')

model = UNet3D(in_channels=opt_model['in_chans'], out_channels=opt_model['out_chans'], final_sigmoid=False)
model.load_state_dict(torch.load(opt['init']['model_dir']))
model.to(device)

def train(epoch):
    model.train()
    average_loss = 0
    average_loss_0 = 0
    average_loss_1 = 0
    average_loss_2 = 0

    for step, (img, pair) in enumerate(train_loader):
        img, pair = img.to(device), pair.to(device)
        output_pair = model(pair)
        loss_0 = loss_l0(output_pair, img, epoch)
        loss_1 = loss_l1(output_pair, img)
        loss_2 = loss_l2(output_pair, img)
        loss = loss_1*opt['train']['loss_l1_weight']+loss_2*opt['train']['loss_l2_weight']+loss_0*opt['train']['loss_l0_weight']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if step == 0:
        #     pair_show = output_pair.detach().cpu().numpy().squeeze()
        #     img_show = img.detach().cpu().numpy().squeeze()
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(img_show[0][0, :, :], cmap='gray')
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(pair_show[0][0, :, :], cmap='gray')
        #     plt.show()

        average_loss += loss.item()
        average_loss_1 += loss_1.item()
        average_loss_2 += loss_2.item()
        average_loss_0 += loss_0.item()
        # print(
        #     '\rTrain Epoch: {}\t Train Loss:{:.6f} Average Loss: {:.3f} Average Loss_1: {:.6f} Average Loss_2: {:.6f}'.format(
        #         epoch, loss.item(),
        #         average_loss / (step + 1), average_loss_1 / (step + 1), average_loss_2 / (step + 1)),end='')
        print(
            '\rTrain Epoch: {}\t Train Loss:{:.6f} Average Loss: {:.3f} Average Loss_1: {:.6f} Average Loss_2: {:.6f} Average Loss_0: {:.6f}'.format(
                epoch, loss.item(),
                average_loss / (step + 1), average_loss_1 / (step + 1), average_loss_2 / (step + 1), average_loss_0 / (step + 1)), end='')

def eval(epoch):
    model.eval()
    output_pair_patches_list = []
    print('\nEval Epoch: {} Start visualizing volume...'.format(epoch+1))
    with torch.no_grad():
        for step, img in enumerate(eval_loader):
            img = img.to(device)
            output_pair = model(img)
            output_pair_patches_list.append(output_pair)
    reconstructed_volume = reconstruct_from_patches(output_pair_patches_list, opt_rec['original_shape'], opt_dataset
    ['train']['patch_size_x'], opt_dataset['train']['patch_size_y'], opt_dataset['train']['patch_size_z'], opt_rec['start_positions'], opt_rec['img_mean'])

    os.makedirs(os.path.join(opt['init']['save_path'], '{}'.format(opt_rec['img_name']),'eval'), exist_ok=True)
    tiff.imwrite(os.path.join(opt['init']['save_path'],'{}'.format(opt_rec['img_name']),'eval', 'reconstructed_{}_epoch_{}.tif'.format(opt_rec['img_name'],epoch+1)), reconstructed_volume)
    print('Eval Epoch: {} Reconstructed volume saved successfully!'.format(epoch+1))
    #save trained model
    os.makedirs(os.path.join(opt['init']['save_path'], '{}'.format(opt_rec['img_name']),'pth'), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(opt['init']['save_path'], '{}'.format(opt_rec['img_name']), 'pth', 'model_epoch_{}.pth'.format(epoch+1)))
    print('Eval Epoch: {} Model saved successfully!'.format(epoch+1))
def predict(epoch):
    model.eval()
    output_pair_patches_list = []
    print('Start visualizing volume...')
    with torch.no_grad():
        for step, img in enumerate(eval_loader):
            img = img.to(device)
            output_pair = model(img)
            output_pair_patches_list.append(output_pair)
            #progress bar
            print('\rProgress: [{}/{}]'.format(step+1, len(eval_loader)), end='')
    # reconstructed_volume = reconstruct_from_patches(output_pair_patches_list, opt_rec['original_shape'], opt_dataset
    # ['train']['patch_size_x'], opt_dataset['train']['patch_size_y'], opt_dataset['train']['patch_size_z'], opt_rec['start_positions'], opt_rec['img_mean'])

    reconstructed_volume = reconstruct_from_trim_patches(output_pair_patches_list, opt_rec['original_shape'], opt_dataset
    ['train']['patch_size_x'], opt_dataset['train']['patch_size_y'], opt_dataset['train']['patch_size_z'], opt_rec['start_positions'], opt_rec['img_mean'])

    os.makedirs(os.path.join(opt['init']['save_path'], '{}'.format(opt_rec['img_name'])), exist_ok=True)
    tiff.imwrite(os.path.join(opt['init']['save_path'],'{}'.format(opt_rec['img_name']), 'reconstructed_{}.tif'.format(opt_rec['img_name'])), reconstructed_volume)
    print('\rReconstructed volume saved successfully!')

if __name__ == '__main__':

    if opt['init']['finetune'] is True:
        print('Start finetuning')
        seed = random.randint(1, 10000)
        print('Random seed: {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        optimizer = Adam(model.parameters(), lr=opt_train['optimizer_lr'],
                         betas=(opt_train['optimizer_beta1'], opt_train['optimizer_beta2']))
        # scheduler = CosineAnnealingLR(optimizer, T_max=1000)
        # loss_char = CharbonnierLoss(opt_train['charbonnier_eps']).to(device)
        # loss_l1 = torch.nn.SmoothL1Loss()
        loss_l0 = L0_Loss(total_epoch=opt['init']['epochs'])
        loss_l1 = torch.nn.L1Loss()
        loss_l2 = torch.nn.MSELoss()
        loss_l0.to(device)
        loss_l1.to(device)
        loss_l2.to(device)

        train_set = Noisy_volume_dataset(opt_dataset['train'], args.config, is_train=True)
        train_loader = data.DataLoader(train_set,
                                       batch_size=opt_dataset['train']['dataloader_batch_size'],
                                       shuffle=opt_dataset['train']['dataloader_shuffle'])

        eval_set = Noisy_volume_dataset(opt_dataset['train'], args.config, is_train=False)
        eval_loader = data.DataLoader(eval_set, batch_size=1, shuffle=False)

        opt = load_yaml(args.config)
        opt_rec = opt['reconstruction']
        for epoch in range(opt['init']['epochs']):
            train(epoch)
            if opt['eval']['eval_epoch'] is not None and (epoch + 1) % opt['eval']['eval_epoch'] == 0:
                eval(epoch)

    else:
        eval_set = Noisy_volume_dataset(opt_dataset['train'], args.config, is_train=False)
        eval_loader = data.DataLoader(eval_set, batch_size=1, shuffle=False)
        opt = load_yaml(args.config)
        opt_rec = opt['reconstruction']
        predict(eval_loader)