import os
import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset
from PIL import Image

class CGIARDataset(Dataset):
    """ CGIAR Dataset """

    def __init__(self, meta_data, root_dir, gt_class, cache_data=True, transform=None):

        df = pd.read_csv(meta_data)
        self.drone_imgs_list = df.folder.to_list()
        self.root_dir = root_dir
        self.transform = transform
        self.cache_data = cache_data
        self.gt_class = gt_class

        self.imgs_path = np.concatenate([glob.glob(os.path.join(self.root_dir, self.drone_imgs_list[i], gt_class, 'patches', '*.png')) for i in range(len(self.drone_imgs_list))])
        print(len(self.imgs_path))
        # print(self.imgs_path)

        if cache_data:
            dataset_imgs = []
            dataset_gt = []
            for data in self.imgs_path:
                aux = np.array(Image.open(data))
                #if (aux.shape[0] != 128) or (aux.shape[1] != 128):
                #    print(data)
                dataset_imgs.append(np.array(Image.open(data)))
                dataset_gt.append(np.array(Image.open(data.replace('patches','masks').replace('patch','mask')).convert('1')))

            print(len(dataset_imgs), len(dataset_gt))    
            self.dataset_imgs = dataset_imgs.copy()
            self.dataset_gt = dataset_gt.copy()

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        
        '''
        Load patch and mask arrays:
        - images come from a single WSI.
        - mask --> 1 for plaques, 0 for background.
        '''
        if self.cache_data:
            image = self.dataset_imgs[idx]
            gt_img = self.dataset_gt[idx]
        else:
            image = np.array(Image.open(self.imgs_path[idx]))
            gt_img = np.array(Image.open(self.imgs_path[idx].replace('patches','masks').replace('patch','mask')).convert('1'))

        sample = {'image': image, 'gt': gt_img}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample['image'], sample['gt']
    
    '''scalar 
    def find_max(im_pths):
    
    
    minimo_pixel=[]
    maximo_pixel=[]
    size=len(im_pths) 

    for i in im_pths:
        img = np.load(str(i))
           
        minimo_pixel.append(np.min(img))
        maximo_pixel.append(np.max(img))

    return   np.min(minimo_pixel),np.max(maximo_pixel), size
        

    '''
