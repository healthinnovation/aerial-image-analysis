import os
import gc
import cv2
import math
import time
import glob
import torch
import logging
import argparse
import threading
import numpy as np
import pandas as pd
import multiprocessing
import tifffile as tiff
import torchvision.transforms.functional as F

from PIL import Image
from operator import itemgetter
from torchvision.ops import masks_to_boxes

Image.MAX_IMAGE_PIXELS = None

# -------------- #
# Util functions #
# -------------- #
def generate_coords_indices(idx, PATCHSIZE, OVERLAP, img_size):
    row_idx = list(range(idx[1], idx[1] + math.ceil((idx[3]-idx[1])/PATCHSIZE)*PATCHSIZE, int(PATCHSIZE*(1-OVERLAP))))
    col_idx = list(range(idx[0], idx[0] + math.ceil((idx[2]-idx[0])/PATCHSIZE)*PATCHSIZE, int(PATCHSIZE*(1-OVERLAP))))

    # Check if last coordinates will generate regions outside of the image.
    if (row_idx[-1] + PATCHSIZE) > img_size[0]:
        row_idx.pop(-1) # remove last item of row indices to avoid extracting patches outside the big image.
    
    if (col_idx[-1] + PATCHSIZE) > img_size[1]:
        col_idx.pop(-1) # remove last item of col indices to avoid extracting patches outside the big image.

    coordinates = zip(row_idx, col_idx)
    return coordinates

def find_ROIs_in_mask(gt):
    # # ------------------------------------------------------------------------------ #
    # # Extract all regions annotated in the image -> instance separation/segmentation #
    # # ------------------------------------------------------------------------------ #
    # _, markers = cv2.connectedComponents(gt)
    # mask_tensor = F.to_tensor(markers)#.to('cuda:0') # mask to gpu
    # obj_ids = torch.unique(mask_tensor)
    # obj_ids = obj_ids[1:]
    # masks = mask_tensor == obj_ids[:, None, None] # masks containing objects detected

    # # --------------------------------------------------------- #
    # # Extract the coordinates of each instance in the big image #
    # # --------------------------------------------------------- #
    # boxes = masks_to_boxes(masks).numpy().astype(int) # (xmin, ymin, xmax, ymax) -> (cmin, rmin, cmax, rmax)
    # # print(f'ROIs detected in drone image:', type(boxes), boxes.dtype, boxes.shape, boxes)

    # del masks
    # gc.collect()

    # return boxes

    # Multiple objects
    contours = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # i = 0
    boxes = []
    for cntr in contours:
        x1,y1,w,h = cv2.boundingRect(cntr)
        x2 = x1+w
        y2 = y1+h
        # print("Object:", i+1, "x1:", x1, "x2:", x2, "y1:", y1, "y2:", y2)
        # i += 1
        boxes.append([x1, y1, x2, y2])

    return np.array(boxes)

def generate_patches(img, gt, boxes, PATCHSIZE, OVERLAP, THRESHOLD):
    # ------------------------- #
    # Extract potential patches #
    # ------------------------- #
    potential_patches = []
    for idx in boxes:
        coordinates = generate_coords_indices(idx, PATCHSIZE, OVERLAP, img.shape)
        for i, (row_idx, col_idx) in enumerate(coordinates):
            aux_patch_img = img[row_idx:row_idx+PATCHSIZE, col_idx:col_idx+PATCHSIZE, :]
            aux_patch_gt = gt[row_idx:row_idx+PATCHSIZE, col_idx:col_idx+PATCHSIZE]
            
            # ------------------ #
            # Validating patches #
            # ------------------ #
            if (np.sum(aux_patch_gt) >= int(PATCHSIZE * PATCHSIZE * THRESHOLD)) and aux_patch_img.shape[0] == 128 and aux_patch_img.shape[1] == 128:
                sample = {'image': aux_patch_img, 'gt': aux_patch_gt, 'coordinates': (row_idx, col_idx), 'gt_perc': np.sum(aux_patch_gt)/int(PATCHSIZE * PATCHSIZE)}
                potential_patches.append(sample) # potential_patches[#elements]['image'/'gt'/'coordinates']

    return potential_patches

# ----------------------------------- #
# Saving patches and creating folders #
# ----------------------------------- #
def create_output_folders(OUTPUTDIR, file):
    
    mask_labels = ['vegetated', 'non_vegetated', 'buildings', 'tillage', 'crops', 'roads']

    for mask_label in mask_labels:
        os.makedirs(os.path.join(OUTPUTDIR, file, mask_label, 'patches'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUTDIR, file, mask_label, 'masks'), exist_ok=True)

def save_patches(potential_patches, outfile_dir):
    for k, sample in enumerate(potential_patches):
        img, gt = sample['image'], sample['gt']
        patch = Image.fromarray(img)
        mask = Image.fromarray(gt).point(lambda i: i * 255)

        patch_name = os.path.join(outfile_dir, f'patches/patch_{k:04}.png')
        mask_name = os.path.join(outfile_dir, f'masks/mask_{k:04}.png')
        patch.save(patch_name)
        mask.save(mask_name)

def save_statistics(potential_patches, outfile_dir):
    coords = list(map(itemgetter('coordinates'), potential_patches))
    gt_area = list(map(itemgetter('gt_perc'), potential_patches))

    csv_info = outfile_dir.split('/')
    csv_dir = outfile_dir.split(f'{csv_info[-2]}/{csv_info[-1]}')[0]
    df = pd.DataFrame({'drone_image': csv_info[-2], 'gt_class': csv_info[-1], 'coodinates': coords, 'gt_area_percentage': gt_area})
    df.to_csv(os.path.join(csv_dir, 'statistics_drone_imgs.csv'), mode='a', index=False, header=False)

    logging.info(f'    Processed image {csv_info[-2]} >>> Number of patches ({csv_info[-1]}): {len(potential_patches)}')
    # logging.info(f'        Number of patches ({csv_info[-1]}): {len(potential_patches)}')


# ----------------------------- #
# Parallel processing functions #
# ----------------------------- #
def process_mask_thread(img, mask_path, PATCHSIZE, OVERLAP, THRESHOLD, OUTPUT_PATH):
    gt = tiff.imread(mask_path)
    boxes = find_ROIs_in_mask(gt)
    patches = generate_patches(img, gt, boxes, PATCHSIZE, OVERLAP, THRESHOLD)

    outfile_dir = None

    # Select where to save patches based on the mask's name
    mask_name = mask_path.split('/')[-1]
    if (mask_name.find('Vegetated') > 0):
        outfile_dir = os.path.join(OUTPUT_PATH, 'vegetated')

    if (mask_name.find('Non_Vegetated') > 0):
        outfile_dir = os.path.join(OUTPUT_PATH, 'non_vegetated')

    if (mask_name.find('Buildings') > 0):
        outfile_dir = os.path.join(OUTPUT_PATH, 'buildings')

    if (mask_name.find('Tillage') > 0):
        outfile_dir = os.path.join(OUTPUT_PATH, 'tillage')

    if (mask_name.find('Crops') > 0):
        outfile_dir = os.path.join(OUTPUT_PATH, 'crops')

    if (mask_name.find('Roads') > 0):
        outfile_dir = os.path.join(OUTPUT_PATH, 'roads')

    if outfile_dir is None:
        logging.info(f'[ERROR] MASK NOT FOUND: {mask_path}')

    save_patches(patches, outfile_dir)
    save_statistics(patches, outfile_dir)

    del gt, patches
    gc.collect()

def extract_patches_parallel(drone_imgs, PATCHSIZE, OVERLAP, DATADIR, OUTPUTDIR):
    '''A thread will launch the process of patch generation using one of the masks files in the drone image folder'''

    # threads_list = []
    for file in drone_imgs: # launch a series of threads for each drone image
        image_path = sorted(glob.glob(os.path.join(DATADIR, file, 'I_*.tif')))
        masks_path = sorted(glob.glob(os.path.join(DATADIR, file, 'M_*.tif')))
        # print(image_path, masks_path)

        create_output_folders(OUTPUTDIR, file)
        OUTPUT_PATH = os.path.join(OUTPUTDIR, file)

        try:
            img = tiff.imread(image_path[0]) # reading the image as it is going to be used by all threads
            img = img[:,:,:3]
        except:
            logging.info(f'Error in image: {image_path[0]}')
            print(image_path)
        
        # Uncomment this for threading!
        threads_list = []
        for mask_path in masks_path:
            thread_mask = threading.Thread(target=process_mask_thread, args=(img, mask_path, PATCHSIZE, OVERLAP, THRESHOLD, OUTPUT_PATH))
            threads_list.append(thread_mask)

        for thread_idx in threads_list: # all threads start at "the same time"
            thread_idx.start()

        for thread_idx in threads_list: # wait for all threads as they process the masks for a single drone image
            thread_idx.join()

        # # This is sequential code >> only one thread per processor is executing.
        # # Comment this for threading!
        # for mask_path in masks_path:
        #     mask_name = mask_path.split('/')[-1]
        #     print(f'        Processing mask {mask_name}')
        #     process_mask_thread(img, mask_path, PATCHSIZE, OVERLAP, THRESHOLD, OUTPUT_PATH)

# ------------ #
# Main program #
# ------------ #
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    
    # ------------------------------- #
    # Parameters for patch generation #
    # ------------------------------- #
    DATADIR = args.data_dir
    OUTPUTDIR = args.output_dir
    PATCHSIZE = args.patch_size
    OVERLAP = args.overlap # patch overlap
    THRESHOLD = args.threshold # minimum percentage of 1s in the mask (patch) to be considered for processing
    NUM_PROC = args.num_workers # number of processors in PC


    drone_imgs = sorted(os.listdir(DATADIR)) # list of all drones images in main directory (input by user)

    # --------------- #
    # Create a logger #
    # --------------- #
    logging.basicConfig(filename=os.path.join(OUTPUTDIR, 'logfile.log'), filemode='w', 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    
    jobs = []
    for i in range(NUM_PROC):
        imgs_per_proc = math.ceil(len(drone_imgs)/NUM_PROC)
        logging.info(f'Processor {i}: Drone images {drone_imgs[i*imgs_per_proc:(i+1)*imgs_per_proc]}')
        process = multiprocessing.Process(target=extract_patches_parallel, args=(drone_imgs[i*imgs_per_proc:(i+1)*imgs_per_proc], PATCHSIZE, OVERLAP, DATADIR, OUTPUTDIR))
        jobs.append(process)

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()