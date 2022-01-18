import os
import argparse
import yaml
from addict import Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import wandb

from trainer import Trainer
from dataset import CGIARDataset
from transforms import ToTensor, Rescale

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_yaml", type=str, default="configs/train.yaml")
    parser.add_argument("--data_dir", type=str, default="./data_patches")
    parser.add_argument("--split_dir", type=str, default="./data_patches")
    parser.add_argument("--load_limit", type=int, default=-1)
    args = parser.parse_args()

    conf = Dict(yaml.safe_load(open(args.train_yaml, "r")))
    
    wandb.init(project="test-project", entity="cgiar")
    wandb.config.update(conf)

    train_file = os.path.join(args.split_dir, 'train_df.csv')
    dev_file = os.path.join(args.split_dir, 'dev_df.csv')
    train_dataset = CGIARDataset(meta_data=train_file,
                                root_dir=args.data_dir,
                                transform=transforms.Compose([
                                Rescale(512), ToTensor()]))
    dev_dataset = CGIARDataset(meta_data=dev_file,
                              root_dir=args.data_dir,
                              transform=transforms.Compose([
                              Rescale(512), ToTensor()]))

    if args.load_limit == -1:
      sampler, shuffle = None, True
    else:
      sampler, shuffle = SubsetRandomSampler(range(args.load_limit)), False

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=shuffle, num_workers=1, sampler=sampler)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size,
                            shuffle=shuffle, num_workers=1, sampler=sampler)

    loaders = {'train': train_dataloader, 'val': dev_dataloader}

    trainer = Trainer(model_opts=conf.model_opts, loaders=loaders)
    trainer.train(args.epochs)
