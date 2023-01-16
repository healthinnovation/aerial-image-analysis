import argparse

from torch.utils.data import DataLoader
from torchvision import transforms

from src.lib.dataset import CGIARDataset
from src.lib.transforms import ToTensor, Rescale

import os
import pandas as pd
import glob
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../dataset_128x128")
    args = parser.parse_args()

    transformed_CGIAR_dataset = CGIARDataset(meta_data='data/experiment_crops_000/test_00.csv',
                                        root_dir=args.data_dir,
                                        gt_class='crops',
                                        cache_data=True,
                                        transform=transforms.Compose([
                                        Rescale(128),
                                        ToTensor()
                                        ]))

    # transformed_StratifIAD_dataset = stratifiadDataset(meta_data='data/experiment_001/train_00_cv_00.csv',
    # 									root_dir=args.data_dir,
    #                                     normalization='macenko')

    dataloader = DataLoader(transformed_CGIAR_dataset, batch_size=4,
                            shuffle=False, num_workers=4)


    ###################################################################
    # To test if stratifiadDataset and Dataloader is working properly #
    ###################################################################
    for i_batch, sample_batched in enumerate(dataloader):
        images, gt = sample_batched
        print(i_batch, images.size(), gt.size())