import yaml
import torch
import itertools

import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss

def define_optim_params(model):
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
        else:
            print('Params [{:s}] will not optimize.'.format(k))
    return optim_params

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        opt = yaml.safe_load(file)
    return opt

def update_yaml(file_path, new_data):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file) or {}
    except FileNotFoundError:
        data = {}

    # Update data
    data.update(new_data)

    with open(file_path, 'w') as file:
        yaml.dump(data, file)

def get_options(opt):
    opt_train = opt['train']
    opt_eval = opt['model']
    opt_dataset = opt['datasets']

    return opt_train, opt_eval, opt_dataset

def reconstruct_from_patches(patches, original_shape, patch_size_x, patch_size_y, patch_size_z, start_positions, img_mean):
    """
    Reconstruct the full 3D image from its patches, considering padding.

    Args:
        patches: List of patches.
        original_shape: The shape of the original 3D stack before padding.
        patch_size_x: patch height.
        patch_size_y: patch width.
        patch_size_z: patch depth (z).
        over_lap: overlap rate.
    Returns:
        Reconstructed 3D image.
    """
    # Adjusted shape after considering padding
    padded_shape = (original_shape[0], original_shape[1], original_shape[2])
    patches = [patch.cpu().numpy().squeeze() if torch.is_tensor(patch) else patch for patch in patches]
    # Initialize an empty array to hold the reconstructed image
    reconstructed = np.zeros(padded_shape, dtype=patches[0].dtype)

    # Create a counter array to average the overlapping patches
    count = np.zeros(padded_shape, dtype=np.float32)

    patch_index = 0
    for z_start in start_positions[0]:
        for h_start in start_positions[1]:
            for w_start in start_positions[2]:
                if patch_index < len(patches):  # Ensure we do not go out of bounds
                    patch = patches[patch_index]
                    reconstructed[z_start:z_start + patch_size_z, h_start:h_start + patch_size_y, w_start:w_start + patch_size_x] += patch
                    count[z_start:z_start + patch_size_z, h_start:h_start + patch_size_y, w_start:w_start + patch_size_x] += 1
                    patch_index += 1
    # Average the overlapping areas
    reconstructed /= count
    reconstructed + img_mean
    return reconstructed

def reconstruct_from_slices(slices):
    reconstructed = np.stack(slices, axis=1)
    return reconstructed

def reconstruct_from_trim_patches(patches, original_shape, patch_size_x, patch_size_y, patch_size_z, start_positions, img_mean, over_lap = 0.25, trim_rate = 0.1):
    """
    Reconstruct the full 3D image from its patches, trimming the edges of non-boundary patches.

    Args:
        patches: List of patches.
        original_shape: The shape of the original 3D stack before padding.
        patch_size_x, patch_size_y, patch_size_z: Dimensions of each patch.
        start_positions: Start positions for each patch.
        img_mean: Mean image value for normalization.
    Returns:
        Reconstructed 3D image.
    """

    padded_shape = (original_shape[0], original_shape[1], original_shape[2])
    patches = [patch.cpu().numpy().squeeze() if torch.is_tensor(patch) else patch for patch in patches]
    # Initialize an empty array to hold the reconstructed image
    reconstructed = np.zeros(padded_shape, dtype=patches[0].dtype)

    # Create a counter array to average the overlapping patches
    count = np.zeros(padded_shape, dtype=np.float32)

    edge_trim_w = int(np.ceil(over_lap*patch_size_x*trim_rate))
    edge_trim_h = int(np.ceil(over_lap*patch_size_y*trim_rate))
    edge_trim_z = int(np.ceil(over_lap*patch_size_z*trim_rate))

    patch_index = 0
    for z_start in start_positions[0]:
        for h_start in start_positions[1]:
            for w_start in start_positions[2]:
                if patch_index < len(patches):
                    patch = patches[patch_index]

                    # Determine the edges to trim based on the boundary status
                    trim_z_start = edge_trim_z if z_start > 0 else 0
                    trim_h_start = edge_trim_h if h_start > 0 else 0
                    trim_w_start = edge_trim_w if w_start > 0 else 0
                    trim_z_end = -edge_trim_z if (z_start + patch_size_z) < original_shape[0] else None
                    trim_h_end = -edge_trim_h if (h_start + patch_size_y) < original_shape[1] else None
                    trim_w_end = -edge_trim_w if (w_start + patch_size_x) < original_shape[2] else None

                    # Trim the patch
                    patch = patch[trim_z_start:trim_z_end, trim_h_start:trim_h_end, trim_w_start:trim_w_end]

                    reconstructed[z_start + trim_z_start:z_start +trim_z_start + patch.shape[0],
                    h_start+trim_h_start:h_start+trim_h_start + patch.shape[1],
                    w_start+trim_w_start:w_start+trim_w_start + patch.shape[2]] += patch

                    count[z_start + trim_z_start:z_start +trim_z_start + patch.shape[0],
                    h_start+trim_h_start:h_start+trim_h_start + patch.shape[1],
                    w_start+trim_w_start:w_start+trim_w_start + patch.shape[2]] += 1
                    # plt.imshow(reconstructed[0, :, :], cmap='gray')
                    # plt.show()
                    patch_index += 1

    count[count == 0] = 1  # Avoid division by zero
    reconstructed /= count
    return reconstructed
