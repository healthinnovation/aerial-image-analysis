import os
import cv2
import math
import time
import glob
import torch
import argparse
import threading
import numpy as np
import pandas as pd
import multiprocessing
import tifffile as tiff
import torchvision.transforms.functional as F

from PIL import Image
from torchvision.ops import masks_to_boxes

Image.MAX_IMAGE_PIXELS = None

# -------------- #
# Util functions #
# -------------- #
def generate_coords_indices(idx, PATCHSIZE, OVERLAP, img_size):
    row_idx = list(range(idx[1], idx[1] + math.ceil((idx[3]-idx[1])/PATCHSIZE)*PATCHSIZE, int(PATCHSIZE*(1-OVERLAP))))
    col_idx = list(range(idx[0], idx[0] + math.ceil((idx[2]-idx[0])/PATCHSIZE)*PATCHSIZE, int(PATCHSIZE*(1-OVERLAP))))
    print("row_idx")
    print(row_idx)

    print("col_idx")
    print(col_idx)

    # Check if last coordinates will generate regions outside of the image.
    if (row_idx[-1] + PATCHSIZE) > img_size[0]:
        print(f"Took last row {row_idx[-1]}")
        row_idx.pop(-1) # remove last item of row indices to avoid extracting patches outside the big image.
    
    if (col_idx[-1] + PATCHSIZE) > img_size[1]:
        print(f"Took last col {col_idx[-1]}")
        col_idx.pop(-1) # remove last item of col indices to avoid extracting patches outside the big image.

    coordinates = zip(row_idx, col_idx)
    return coordinates

def save_patches(potential_patches):
    for sample in potential_patches:
        img, gt, coords = sample['image'], sample['gt'], sample['coordinates']
        patch = Image.fromarray(img)
        mask = Image.fromarray(gt).point(lambda i: i * 255)
        patch.save('output_patch.png')
        mask.save('output_mask.png')

        if img.shape[0] != 128 or img.shape[1] != 128:
            print("shape:", img.shape)

def create_output_folders(OUTPUTDIR, file):
    
    mask_labels = ['vegetated', 'non_vegetated', 'buildings', 'tillage', 'crops', 'roads']

    for mask_label in mask_labels:
        os.makedirs(os.path.join(OUTPUTDIR, file, mask_label), exist_ok=True)

def process_mask_thread(img, mask_path):

    gt = tiff.imread(mask_path)


def extract_patches_parallel(drone_imgs, PATCHSIZE, OVERLAP, DATADIR, OUTPUTDIR):
    '''A thread will launch the process of patch generation using one of the masks files in the drone image folder'''

    threads_list = []
    for file in drone_imgs: # launch a series of threads for each drone image
        print(f'Processing image {file}')
        image_path = sorted(glob.glob(os.path.join(DATADIR, file, 'I_*.tif')))
        masks_path = sorted(glob.glob(os.path.join(DATADIR, file, 'M_*.tif')))
        # print(image_path, masks_path)

        create_output_folders(OUTPUTDIR, file)

        img = tiff.imread(image_path[0]) # reading the image as it is going to be used by all threads
        img = img[:,:,:3]
        
        threads_list = []
        for mask_path in masks_path:
            mask_folder = mask_path.split('/')
            print(mask_folder[-1].find('Crops'))
        #     thread_mask = threading.Thread(target=process_mask_thread, args=(img, mask_path))
        #     threads_list.append(thread_mask)

        # for thread_idx in threads_list: # all threads start at "the same time"
        #     thread_idx.start()

        # for thread_idx in threads_list: # wait for all threads as they process the masks for a single drone image
        #     thread_idx.join()

# ------------ #
# Main program #
# ------------ #
root_dir = '../../civ/20210726006_labels'
image_name = 'I_20210726006-ortho.tif'
mask_name = 'M_Crops_20210726006.tif'

PATCHSIZE = 128
OVERLAP = 0.5 # patch overlap
THRESHOLD = 0.05 # minimum percentage of 1s in the mask (patch) to be considered for processing

# ---------------------------------- #
# Reading the image and the mask tif #
# ---------------------------------- #
tic = time.time()
img = tiff.imread(os.path.join(root_dir, image_name))
print(img.shape)
img = img[:,:,:3]
gt = tiff.imread(os.path.join(root_dir, mask_name))
toc = time.time()
print(f"Total time to read img and mask: {(toc - tic):0.4f} s")

# ------------------------------------------------------------------------------ #
# Extract all regions annotated in the image -> instance separation/segmentation #
# ------------------------------------------------------------------------------ #
_, markers = cv2.connectedComponents(gt)
mask_tensor = F.to_tensor(markers)#.to('cuda:0') # mask to gpu
obj_ids = torch.unique(mask_tensor)
obj_ids = obj_ids[1:]
masks = mask_tensor == obj_ids[:, None, None] # masks containing objects detected


# Multiple objects
contours = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
i = 0
boxes_cv = []
for cntr in contours:
    x1,y1,w,h = cv2.boundingRect(cntr)
    x2 = x1+w
    y2 = y1+h
    # print("Object:", i+1, "x1:", x1, "x2:", x2, "y1:", y1, "y2:", y2)
    i += 1
    boxes_cv.append([x1, y1, x2, y2])


print(np.array(boxes_cv), type(boxes_cv))
print("\n\n")

# --------------------------------------------------------- #
# Extract the coordinates of each instance in the big image #
# --------------------------------------------------------- #
boxes = masks_to_boxes(masks).numpy().astype(int) # (xmin, ymin, xmax, ymax) -> (cmin, rmin, cmax, rmax)
# print(f'ROIs detected in drone image:', type(boxes), boxes.dtype, boxes.shape, boxes)
print(boxes, type(boxes))

# ------------------------- #
# Extract potential patches #
# ------------------------- #
tic = time.time()
potential_patches = []
for idx in boxes:
    coordinates = generate_coords_indices(idx, PATCHSIZE, OVERLAP, img.shape)
    for i, (row_idx, col_idx) in enumerate(coordinates):
        aux_patch_img = img[row_idx:row_idx+PATCHSIZE, col_idx:col_idx+PATCHSIZE, :]
        aux_patch_gt = gt[row_idx:row_idx+PATCHSIZE, col_idx:col_idx+PATCHSIZE]

        if aux_patch_img.shape[0] != 128 or aux_patch_img.shape[1] != 128:
            print("aux shape:", aux_patch_img.shape)
            print(row_idx)
            print(row_idx+PATCHSIZE)
            print(col_idx)
            print(col_idx+PATCHSIZE)            

        # ------------------ #
        # Validating patches #
        # ------------------ #
        if (np.sum(aux_patch_gt) >= int(PATCHSIZE * PATCHSIZE * THRESHOLD)) and aux_patch_img.shape[0] == 128 and aux_patch_img.shape[1] == 128:
            sample = {'image': aux_patch_img, 'gt': aux_patch_gt, 'coordinates': (row_idx, col_idx)}
            potential_patches.append(sample) # potential_patches[#elements]['image'/'gt'/'coordinates']

toc = time.time()
print(f"Total time for patch extraction: {(toc - tic)*1e3:0.4f} ms")
print(f'Number of patches: {len(potential_patches)}')
# print(potential_patches[0]['coordinates'][0])

# ------------ #
# Save patches #
# ------------ #
save_patches(potential_patches)

