import os
import argparse
import numpy as np
import pandas as pd
import glob

<<<<<<< HEAD
root_dir = '../../dataset_512x512/'
gt_class = 'non_vegetated'
=======
root_dir = '../../dataset_256x256/'
gt_class = 'vegetated'
>>>>>>> 42951068e7560a02fbd4c24434948ed3b02e4130

# Process for test:
test_csv = '../data/experiment_'+gt_class+'_000/test_00.csv'
test_drone_imgs = pd.read_csv(test_csv).folder.to_list()
test_imgs_path = np.concatenate([glob.glob(os.path.join(root_dir, test_drone_imgs[i], gt_class, 'patches', '*.png')) for i in range(len(test_drone_imgs))])

#print([os.path.join(root_dir, test_drone_imgs[i], gt_class, 'patches', '*.png') for i in range(len(test_drone_imgs))])
#print(test_imgs_path[:10])


# Process for train/val
dev_drone_imgs = []
dev_csv_00 = '../data/experiment_'+gt_class+'_000/dev_00_cv_00.csv'
dev_csv_01 = '../data/experiment_'+gt_class+'_000/dev_00_cv_01.csv'
dev_csv_02 = '../data/experiment_'+gt_class+'_000/dev_00_cv_02.csv'
dev_drone_imgs = np.concatenate([pd.read_csv(dev_csv_00).folder.to_list(), pd.read_csv(dev_csv_01).folder.to_list(), pd.read_csv(dev_csv_02).folder.to_list()])
dev_imgs_path = np.concatenate([glob.glob(os.path.join(root_dir, dev_drone_imgs[i], gt_class, 'patches', '*.png')) for i in range(len(dev_drone_imgs))])

# Results
print(f'# patches for train/val: {len(dev_imgs_path)}')
print(f'# patches for test: {len(test_imgs_path)}')

