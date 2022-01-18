from __future__ import print_function, division
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import warnings
import sys
import os

warnings.filterwarnings("ignore")
libdir = os.path.dirname(__file__)
sys.path.append(os.path.split(libdir)[0])

from src.dataset import CGIARDataset
from src.transforms import ToTensor, Rescale

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data_patches")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

transformed_dataset = CGIARDataset(meta_data='./data_patches/dev_df.csv',
                    					root_dir=args.data_dir,
                                        transform=transforms.Compose([
                                        Rescale(512),
                                        ToTensor()
                                        ]))

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, **kwargs)


##########################################################
# To test if Dataset and Dataloader are working properly #
##########################################################
for i_batch, sample_batched in enumerate(dataloader):
    images, gt = sample_batched
    print(f'{i_batch}, {images.size()}, {gt.size()}')