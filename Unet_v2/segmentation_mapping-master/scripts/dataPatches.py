#import tifffile as tiff
from pathlib import Path
#import glob
import numpy as np
#import matplotlib.pyplot as plt
import os
import shutil
import tifffile as tiff
from csv import reader

trainCsv_dir = '../../../data_Unetv1/train_df.csv'
testCsv_dir = '../../../data_Unetv1/test_df.csv'

pathOriginImages = '../../../data_Unetv1/images/'
pathOriginMasks = '../../../data_Unetv1/masks/'

pathTestImages = '../../../data_Unetv2/test/images'
pathTestMasks = '../../../data_Unetv2/test/masks'

pathTrainImages = '../../../data_Unetv2/train_val/images'
pathTrainMasks = '../../../data_Unetv2/train_val/masks'

Path(pathTestImages).mkdir(parents=True, exist_ok=True)
Path(pathTestMasks).mkdir(parents=True, exist_ok=True)
Path(pathTrainImages).mkdir(parents=True, exist_ok=True)
Path(pathTrainMasks).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":


# Test Images:

    with open(testCsv_dir, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            print(row[0])
            
            # Images
            img = os.path.join(pathOriginImages, row[0])
            print('origin img:', img)
            new_path_img = os.path.join(pathTestImages, row[0].replace('.tif','.npy'))
            
            print('command img old:', img)             
            print('command img new path:', new_path_img)

            # Tiff to npy
            image_tiff = tiff.imread(img)           
            img_numpy = np.asarray(image_tiff).transpose(2,0,1)
            np.save(new_path_img, img_numpy)
            #shutil.copy(img, new_path_img)
            
            
            # Masks
            mask = os.path.join(pathOriginMasks, row[0].replace('.tif','_mask.tif'))
            
            print('origin mask:', mask)
            new_path_mask = os.path.join(pathTestMasks, row[0].replace('.tif','_mask.npy'))
            
            print('command gt old:', mask)             
            print('command gt new path:', new_path_mask)

            # Tiff to npy
            mask_tiff = tiff.imread(mask)           
            mask_numpy = np.asarray(mask_tiff)
            print(mask_numpy.shape)
            np.save(new_path_mask, mask_numpy)

            #shutil.copy(mask, new_path_mask)
            

# Train:

    with open(trainCsv_dir, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            print(row[0])
            
            # Images
            img = os.path.join(pathOriginImages, row[0])
            
            print('origin img:', img)
            new_path_img = os.path.join(pathTrainImages, row[0].replace('.tif','.npy'))
            
            print('command img old:', img)             
            print('command img new path:', new_path_img)
            #shutil.copy(img, new_path_img)

            # Tiff to npy
            image_tiff = tiff.imread(img)           
            img_numpy = np.asarray(image_tiff).transpose(2,0,1)
            np.save(new_path_img, img_numpy)
            #shutil.copy(img, new_path_img)

            # Masks
            mask = os.path.join(pathOriginMasks, row[0].replace('.tif','_mask.tif'))
            
            print('origin mask:', mask)
            new_path_mask = os.path.join(pathTrainMasks, row[0].replace('.tif','_mask.npy'))

            # Tiff to npy
            mask_tiff = tiff.imread(mask)           
            mask_numpy = np.asarray(mask_tiff)
            print(mask_numpy.shape)
            np.save(new_path_mask, mask_numpy)

            #shutil.copy(mask, new_path_mask)

            print('command gt old:', mask)             
            print('command gt new path:', new_path_mask)
            #shutil.copy(mask, new_path_mask)
