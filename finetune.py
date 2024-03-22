import os
import torch
import argparse

import tifffile as tiff

from torch.utils import data
from model_zoo.network_unet3d import UNet3D
from dataset_n2n_3d import Noisy_volume_dataset
from helper import get_options, load_yaml, reconstruct_from_patches




parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./finetune_config.yaml',
                    help='Path to YAML configuration file.')
parser.add_argument('-o', '--output_dir', type=str, default='./train_config.yaml',
                    help='Path to output image.')
parser.add_argument('-i', '--input_dir', type=str, default='./train_config.yaml',
                    help='Path to input image.')
parser.add_argument('-m', '--model_dir', type=str, default='./train_config.yaml',
                    help='Path to model.')

args = parser.parse_args()
opt = load_yaml(args.config)

opt_train, opt_model, opt_dataset  = get_options(opt)

device = torch.device('cuda' if opt['init']['gpu_ids'] is not None else 'cpu')

model = UNet3D(n_channels=opt_model['in_chans'], n_classes=opt_model['out_chans'])
model.load_state_dict(torch.load(opt_model['pth_path']))
model.to(device)

def eval(epoch):
    model.eval()
    output_pair_patches_list = []
    with torch.no_grad():
        for step, img in enumerate(eval_loader):
            img = img.to(device)
            output_pair = model(img)
            output_pair_patches_list.append(output_pair)

    reconstructed_volume = reconstruct_from_patches(output_pair_patches_list, opt_rec['original_shape'],
                                                    opt_dataset
                                                    ['train']['patch_size_x'], opt_dataset['train']['patch_size_y'],
                                                    opt_dataset['train']['patch_size_z'],
                                                    opt_rec['start_positions'])

    os.makedirs(os.path.join(opt['init']['save_path'], '{}'.format(opt_rec['img_name']), 'eval'),
                exist_ok=True)
    tiff.imwrite(os.path.join(opt['init']['save_path'], '{}'.format(opt_rec['img_name']), 'eval',
                              'reconstructed_{}_epoch_{}.tif'.format(opt_rec['img_name'], epoch + 1)),
                 reconstructed_volume)
    # save trained model
    os.makedirs(os.path.join(opt['init']['save_path'], '{}'.format(opt_rec['img_name']), 'pth'),
                exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(opt['init']['save_path'], '{}'.format(opt_rec['img_name']), 'pth',
                            'model_epoch_{}.pth'.format(epoch + 1)))


if __name__ == '__main__':
    eval_set = Noisy_volume_dataset(opt_dataset['train'], args.config, is_train=False)
    eval_loader = data.DataLoader(eval_set, batch_size=1, shuffle=False)

    opt = load_yaml(opt.config)
    opt_rec = opt['reconstruction']
    eval(eval_loader)
