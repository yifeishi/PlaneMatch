import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import torchvision
import torch.nn as nn
import math
from cropextract import *
from torch.utils.data import Dataset, DataLoader
import scipy.misc as smi

import scipy.io as sio
import scipy.misc as smi


class PlanarPatchDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        rgb_local_name1 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 0])
        rgb_global_name1 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 1])
        depth_local_name1 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 2])
        depth_global_name1 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 3])
        normal_local_name1 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 4])
        normal_global_name1 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 5])
        mask_local_name1 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 6])
        mask_global_name1 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 7])

        rgb_local_name2 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 8])
        rgb_global_name2 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 9])
        depth_local_name2 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 10])
        depth_global_name2 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 11])
        normal_local_name2 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 12])
        normal_global_name2 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 13])
        mask_local_name2 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 14])
        mask_global_name2 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 15])

        rgb_local_name3 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 16])
        rgb_global_name3 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 17])
        depth_local_name3 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 18])
        depth_global_name3 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 19])
        normal_local_name3 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 20])
        normal_global_name3 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 21])
        mask_local_name3 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 22])
        mask_global_name3 = os.path.join(self.root_dir,self.landmarks_frame.ix[idx, 23])
        
        rgb_local_image1 = smi.imread(rgb_local_name1)
        rgb_global_image1 = smi.imread(rgb_global_name1)
        depth_local_image1 = smi.imread(depth_local_name1)
        depth_global_image1 = smi.imread(depth_global_name1)
        normal_local_image1 = smi.imread(normal_local_name1)
        normal_global_image1 = smi.imread(normal_global_name1)
        mask_local_image1 = smi.imread(mask_local_name1)
        mask_global_image1 = smi.imread(mask_global_name1)

        rgb_local_image2 = smi.imread(rgb_local_name2)
        rgb_global_image2 = smi.imread(rgb_global_name2)
        depth_local_image2 = smi.imread(depth_local_name2)
        depth_global_image2 = smi.imread(depth_global_name2)
        normal_local_image2 = smi.imread(normal_local_name2)
        normal_global_image2 = smi.imread(normal_global_name2)
        mask_local_image2 = smi.imread(mask_local_name2)
        mask_global_image2 = smi.imread(mask_global_name2)

        rgb_local_image3 = smi.imread(rgb_local_name3)
        rgb_global_image3 = smi.imread(rgb_global_name3)
        depth_local_image3 = smi.imread(depth_local_name3)
        depth_global_image3 = smi.imread(depth_global_name3)
        normal_local_image3 = smi.imread(normal_local_name3)
        normal_global_image3 = smi.imread(normal_global_name3)
        mask_local_image3 = smi.imread(mask_local_name3)
        mask_global_image3 = smi.imread(mask_global_name3)

        sample = {'rgb_global_image1': rgb_global_image1, 'rgb_global_image2': rgb_global_image2, 'rgb_global_image3': rgb_global_image3,
                  'depth_global_image1': depth_global_image1, 'depth_global_image2': depth_global_image2, 'depth_global_image3': depth_global_image3,
                  'normal_global_image1': normal_global_image1, 'normal_global_image2': normal_global_image2, 'normal_global_image3': normal_global_image3,
                  'mask_global_image1': mask_global_image1, 'mask_global_image2': mask_global_image2, 'mask_global_image3': mask_global_image3,
                  'rgb_local_image1': rgb_local_image1, 'rgb_local_image2': rgb_local_image2, 'rgb_local_image3': rgb_local_image3,
                  'depth_local_image1': depth_local_image1, 'depth_local_image2': depth_local_image2, 'depth_local_image3': depth_local_image3,
                  'normal_local_image1': normal_local_image1, 'normal_local_image2': normal_local_image2, 'normal_local_image3': normal_local_image3,
                  'mask_local_image1': mask_local_image1, 'mask_local_image2': mask_local_image2, 'mask_local_image3': mask_local_image3}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        rgb_global_image1, rgb_global_image2, rgb_global_image3, \
        depth_global_image1, depth_global_image2, depth_global_image3,\
        normal_global_image1, normal_global_image2, normal_global_image3,\
        mask_global_image1, mask_global_image2, mask_global_image3, \
        rgb_local_image1, rgb_local_image2, rgb_local_image3, \
        depth_local_image1, depth_local_image2, depth_local_image3,\
        normal_local_image1, normal_local_image2, normal_local_image3,\
        mask_local_image1, mask_local_image2, mask_local_image3,= \
        sample['rgb_global_image1'], sample['rgb_global_image2'], sample['rgb_global_image3'], \
        sample['depth_global_image1'], sample['depth_global_image2'], sample['depth_global_image3'], \
        sample['normal_global_image1'], sample['normal_global_image2'], sample['normal_global_image3'], \
        sample['mask_global_image1'], sample['mask_global_image2'], sample['mask_global_image3'], \
        sample['rgb_local_image1'], sample['rgb_local_image2'], sample['rgb_local_image3'], \
        sample['depth_local_image1'], sample['depth_local_image2'], sample['depth_local_image3'], \
        sample['normal_local_image1'], sample['normal_local_image2'], sample['normal_local_image3'], \
        sample['mask_local_image1'], sample['mask_local_image2'], sample['mask_local_image3']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        rgb_global_image1 = rgb_global_image1.astype(np.float).transpose((2, 0, 1)) - 128
        rgb_global_image2 = rgb_global_image2.astype(np.float).transpose((2, 0, 1)) - 128
        rgb_global_image3 = rgb_global_image3.astype(np.float).transpose((2, 0, 1)) - 128

        scl = 1. / 4000. * 255.
        depth_global_image1 = np.multiply(depth_global_image1, scl).astype(np.float) - 128
        depth_global_image2 = np.multiply(depth_global_image2, scl).astype(np.float) - 128
        depth_global_image3 = np.multiply(depth_global_image3, scl).astype(np.float) - 128

        normal_global_image1 = normal_global_image1.astype(np.float).transpose((2, 0, 1)) - 128
        normal_global_image2 = normal_global_image2.astype(np.float).transpose((2, 0, 1)) - 128
        normal_global_image3 = normal_global_image3.astype(np.float).transpose((2, 0, 1)) - 128

        mask_global_image1 = mask_global_image1.astype(np.float) - 128
        mask_global_image2 = mask_global_image2.astype(np.float) - 128
        mask_global_image3 = mask_global_image3.astype(np.float) - 128
       
        rgb_local_image1 = rgb_local_image1.astype(np.float).transpose((2, 0, 1)) - 128
        rgb_local_image2 = rgb_local_image2.astype(np.float).transpose((2, 0, 1)) - 128
        rgb_local_image3 = rgb_local_image3.astype(np.float).transpose((2, 0, 1)) - 128
        
        scl = 1. / 4000. * 255.
        depth_local_image1 = np.multiply(depth_local_image1, scl).astype(np.float) - 128
        depth_local_image2 = np.multiply(depth_local_image2, scl).astype(np.float) - 128
        depth_local_image3 = np.multiply(depth_local_image3, scl).astype(np.float) - 128

        normal_local_image1 = normal_local_image1.astype(np.float).transpose((2, 0, 1)) - 128
        normal_local_image2 = normal_local_image2.astype(np.float).transpose((2, 0, 1)) - 128
        normal_local_image3 = normal_local_image3.astype(np.float).transpose((2, 0, 1)) - 128
        
        mask_local_image1 = mask_local_image1.astype(np.float) - 128
        mask_local_image2 = mask_local_image2.astype(np.float) - 128
        mask_local_image3 = mask_local_image3.astype(np.float) - 128
        
        return {'rgb_global_image1': torch.from_numpy(rgb_global_image1), 'rgb_global_image2': torch.from_numpy(rgb_global_image2), 'rgb_global_image3': torch.from_numpy(rgb_global_image3), 
                'depth_global_image1': torch.from_numpy(depth_global_image1), 'depth_global_image2': torch.from_numpy(depth_global_image2), 'depth_global_image3': torch.from_numpy(depth_global_image3),
                'normal_global_image1': torch.from_numpy(normal_global_image1), 'normal_global_image2': torch.from_numpy(normal_global_image2), 'normal_global_image3': torch.from_numpy(normal_global_image3),
                'mask_global_image1': torch.from_numpy(mask_global_image1), 'mask_global_image2': torch.from_numpy(mask_global_image2), 'mask_global_image3': torch.from_numpy(mask_global_image3),
                'rgb_local_image1': torch.from_numpy(rgb_local_image1), 'rgb_local_image2': torch.from_numpy(rgb_local_image2), 'rgb_local_image3': torch.from_numpy(rgb_local_image3), 
                'depth_local_image1': torch.from_numpy(depth_local_image1), 'depth_local_image2': torch.from_numpy(depth_local_image2), 'depth_local_image3': torch.from_numpy(depth_local_image3),
                'normal_local_image1': torch.from_numpy(normal_local_image1), 'normal_local_image2': torch.from_numpy(normal_local_image2), 'normal_local_image3': torch.from_numpy(normal_local_image3),
                'mask_local_image1': torch.from_numpy(mask_local_image1), 'mask_local_image2': torch.from_numpy(mask_local_image2), 'mask_local_image3': torch.from_numpy(mask_local_image3)}
