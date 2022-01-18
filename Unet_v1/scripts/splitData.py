import os
import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dev_perc", type=float, default=0.1)
    parser.add_argument("--test_perc", type=float, default=0.1)
    parser.add_argument("--dataset_csv", type=str, default="../data_patches/patch_dataset.csv")

    args = parser.parse_args()

    tiles = pd.read_csv(args.dataset_csv)
    tiles = tiles.sample(len(tiles), random_state=1)
    
    dev_perc = args.dev_perc
    test_perc = args.test_perc
    train_perc = 1 - (dev_perc + test_perc)
    
    split_1 = int(train_perc * len(tiles))
    split_2 = int((1-dev_perc) * len(tiles))
    
    train = tiles[:split_1]
    dev = tiles[split_1:split_2]
    test = tiles[split_2:]
    
    train.to_csv('../data_patches/train_df.csv', index=False)
    dev.to_csv('../data_patches/dev_df.csv', index=False)
    test.to_csv('../data_patches/test_df.csv', index=False)

    print(f'Total patches: {len(tiles)}')
    print(f'Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}')