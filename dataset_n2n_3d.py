import os
import yaml
import torch
import random

import tifffile as tiff
import numpy as np

from torch.utils import data
from torchvision import transforms as T

import matplotlib.pyplot as plt

class Noisy_volume_dataset(data.Dataset):
    def __init__(self, opt, yaml_path, is_train=True):
        super(Noisy_volume_dataset, self).__init__()
        self.train = is_train
        self.opt = opt
        self.yaml_path = yaml_path
        self.transform = T.Compose([T.ToTensor()])
        self.normalize = opt['normalize']
        self.path = opt['data_path']
        self.image_paths = self.get_image_paths(self.path)
        self.over_lap = opt['patch_overlap']
        self.data = self.read_tiff(self.image_paths[0])
        self.z, self.h, self.w = self.data.shape
        # self.data, self.padding_size = self.add_padding(self.data, self.opt['patch_size_x'], self.opt['patch_size_y'], self.opt['patch_size_z'], self.opt['patch_overlap'])
        mean = np.mean(self.data)
        self.data = self.data - mean
        self.img_name = self.image_paths[0].split('\\')[-1].split('.')[0]
        print(f'Loading img: {self.img_name}')
        if self.train:
            self.input_patches, self.target_patches = self.split_into_patches(self.data, self.opt['patch_size_x'],
                                                                              self.opt['patch_size_y'],
                                                                              self.opt['patch_size_z'],
                                                                              self.opt['patch_overlap'], self.train)
        else:
            self.input_patches, self.start_positions = self.split_into_patches(self.data, self.opt['patch_size_x'], self.opt['patch_size_y'],
                                                         self.opt['patch_size_z'], self.opt['patch_overlap'],
                                                         self.train)

            params = {
                'reconstruction': {
                    'original_shape': [self.z, self.h, self.w],
                    # 'padding_size': self.padding_size,
                    'img_mean': round(mean.tolist(), 4),
                    'img_name': self.img_name,
                    'start_positions': self.start_positions
                }
            }
            self.update_yaml(self.yaml_path, params)

    def __getitem__(self, index):
        if self.train:
            input_patch = self.input_patches[index]
            target_patch = self.target_patches[index]
            input_patch, target_patch = self.random_transform(input_patch, target_patch)
            input_patch = torch.from_numpy(np.expand_dims(input_patch.copy(), 0)).float()
            target_patch = torch.from_numpy(np.expand_dims(target_patch.copy(), 0)).float()
            return input_patch, target_patch
        else:
            input_patch = self.input_patches[index].astype(np.float)
            input_patch = torch.from_numpy(np.expand_dims(input_patch, 0)).float()
            return input_patch

    def __len__(self):
        if self.train:
            return len(self.input_patches)
        else:
            return len(self.input_patches)

    def add_noise(self, img):
        noise = torch.randn(img.shape).mul_(self.sigma / 255.0)
        img = img.add_(noise)
        return img

    def read_tiff(self, path):
        return tiff.imread(path)

    def divide_stack(self, img_stack):
        # divide a stack into odd/even frames
        img_stack = self.read_tiff(img_stack)
        odd = img_stack[0::2]
        even = img_stack[1::2]
        if odd.shape[0] != even.shape[0]:
            odd = odd[:-1]
        return odd, even

    def update_yaml(self, file_path, new_data):
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file) or {}
        except FileNotFoundError:
            data = {}

        # Update data
        data.update(new_data)

        with open(file_path, 'w') as file:
            yaml.dump(data, file)

    def get_image_paths(self, path):
        """
        Get a list of image paths from a given directory.

        Args:
            path (str): Directory path to search for images.

        Returns:
            list: List of full paths to images in the given directory.

        Raises:
            FileNotFoundError: If the given path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'The path {path} does not exist!')

        return [os.path.join(path, img) for img in sorted(os.listdir(path))]

    def random_transform(self, input, target):
        """
        Simplified function for data augmentation. Randomly selects a transformation method
        among rotation, flip, or a combination, or applies no transformation.

        Args:
            input, target: The input and target patch before data augmentation.

        Returns:
            input, target: The input and target patch after data augmentation.
        """

        def swap_small_regions(patch1, patch2, region_size=(16, 16)):
            """
            在两个patch中相同位置的较小区域内随机交换像素。

            Args:
                patch1, patch2: 尺寸为[12, 64, 64]的两个patch。
                region_size: 要交换的区域大小，例如(8, 8)。
                num_regions: 要交换的区域数量。

            Returns:
                交换像素后的两个patch。
            """

            z, h, w = patch1.shape
            region_h, region_w = region_size

            # iterate over the patch
            for i in range(0, h, region_h):
                for j in range(0, w, region_w):
                    if random.random() < 0.5:
                        for k in range(z):
                            # swap regions
                            temp = patch1[k, i:i + region_h, j:j + region_w].copy()
                            patch1[k, i:i + region_h, j:j + region_w] = patch2[k, i:i + region_h, j:j + region_w]
                            patch2[k, i:i + region_h, j:j + region_w] = temp

            return patch1, patch2

        def apply_transform(input, target, rotate_times=0, flip=False):
            if flip:
                input = input[:, :, ::-1]
                target = target[:, :, ::-1]

            input = np.rot90(input, rotate_times, (1, 2))
            target = np.rot90(target, rotate_times, (1, 2))
            return input, target

        def swap_pixel(input, target):
            z, h, w = input.shape

            for i in range(z):
                mask = np.random.rand(h, w) > 0.5
                # swap pixels
                temp = np.copy(input[i, :, :])
                input[i, :, :][mask] = target[i, :, :][mask]
                target[i, :, :][mask] = temp[mask]
            return input, target

        p_trans = random.randrange(8)
        rotate_times = p_trans % 4  # 0, 1, 2, or 3 times rotation
        flip = p_trans >= 4  # Flip if p_trans is 4 or greater

        input, target = apply_transform(input, target, rotate_times, flip)

        # if random.random() < 0.5:
        #     input, target = swap_small_regions(input, target)

        if random.random() < 0.5:
            input, target = target, input

        # Random brightness adjustment
        # if random.random() < 0.5:
        #     brightness_delta = 0.1
        #     delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        #     input *= delta
        #     target *= delta

        return input, target

    def split_into_patches(self, image, patch_size_x, patch_size_y, patch_size_z, over_lap, is_train=True):
        """
        Cut 3D stack into patches considering the overlap and different modes (train or eval).

        Args:
            image: 3D stack, shape (z, h, w)
            patch_size_x: Patch height
            patch_size_y: Patch width
            patch_size_z: Patch depth (z)
            over_lap: Overlap rate (0 to 1)
            is_train: Boolean flag to indicate training or evaluation mode

        Returns:
            List includes patches or two lists (input and target) in case of training mode
        """
        z, h, w = image.shape

        # Calculate the stride (step size) considering the overlap
        stride_x = int(patch_size_x * (1 - over_lap))
        stride_y = int(patch_size_y * (1 - over_lap))
        stride_z = int(patch_size_z * (1 - over_lap))

        def calculate_starts(dim_size, patch_size, stride):
            starts = list(range(0, dim_size - patch_size + 1, stride))
            if dim_size - patch_size > starts[-1]:
                starts.append(dim_size - patch_size)
            return starts

        if is_train:
            # Splitting the 3D stack into odd and even frames
            odd_image = image[0::2, :, :]
            even_image = image[1::2, :, :]

            # Adjusting the starts for odd and even images
            z_starts_odd = calculate_starts(odd_image.shape[0], patch_size_z, stride_z)
            z_starts_even = calculate_starts(even_image.shape[0], patch_size_z, stride_z)
            y_starts = calculate_starts(h, patch_size_x, stride_x)
            x_starts = calculate_starts(w, patch_size_y, stride_y)

            patches_input, patches_target = [], []
            for z_start in z_starts_odd:
                for y_start in y_starts:
                    for x_start in x_starts:
                        patch = odd_image[z_start:z_start + patch_size_z, y_start:y_start + patch_size_x,
                                x_start:x_start + patch_size_y]
                        patches_input.append(patch)

            for z_start in z_starts_even:
                for y_start in y_starts:
                    for x_start in x_starts:
                        patch = even_image[z_start:z_start + patch_size_z, y_start:y_start + patch_size_x,
                                x_start:x_start + patch_size_y]
                        patches_target.append(patch)

            return patches_input, patches_target

        else:
            # For evaluation mode, process the whole stack directly
            z_starts = calculate_starts(z, patch_size_z, stride_z)
            y_starts = calculate_starts(h, patch_size_x, stride_x)
            x_starts = calculate_starts(w, patch_size_y, stride_y)

            patches = []
            for z_start in z_starts:
                for y_start in y_starts:
                    for x_start in x_starts:
                        patch = image[z_start:z_start + patch_size_z, y_start:y_start + patch_size_x,
                                x_start:x_start + patch_size_y]
                        patches.append(patch)

            return patches, [z_starts, y_starts, x_starts]