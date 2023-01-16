import os
<<<<<<< HEAD
=======
<<<<<<< HEAD
import torch
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
from src.lib.models.unet import Unet
import torchvision.transforms.functional as F
from scipy.io import savemat

from PIL import Image
from tqdm import tqdm
from patchify import patchify, unpatchify

def generate_patches(drone_img):
    # Pad the entire image to avoid losing border information using patchify
    row_pad = int(np.ceil(drone_img.shape[0]/PATCHSIZE)*PATCHSIZE - drone_img.shape[0])
    col_pad = int(np.ceil(drone_img.shape[1]/PATCHSIZE)*PATCHSIZE - drone_img.shape[1])
    drone_img_padding = np.pad(drone_img, ((0, row_pad), (0, col_pad), (0,0)), 'constant')

    # All patches in a single array of size (img.shape[0]/PATCHSIZE, img.shape[1]/PATCHSIZE, 1, PATCHSIZE, PATCHSIZE, 3)
    patches = patchify(drone_img_padding, (PATCHSIZE, PATCHSIZE, 3), step=PATCHSIZE) 
    patches_loader = np.reshape(patches, (patches.shape[0]*patches.shape[1]*patches.shape[2],patches.shape[3],patches.shape[4],patches.shape[5]))
    print(f'Size of patches (patchify): {patches.shape}, data type: {patches.dtype}')
    print(f'Size of patch loader: {patches_loader.shape}, data type: {patches_loader.dtype}')

    return patches_loader, patches.shape, drone_img_padding.shape


def generate_probability_map_per_class(patches_loader, patches_size, drone_img_padding_size, drone_img_size, PATCHSIZE, weights_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if PATCHSIZE == 512:
        model = Unet(inchannels=3, outchannels=1, net_depth=5) # This is for patchsize 512x512 pixels
    if PATCHSIZE == 256:
        model = Unet(inchannels=3, outchannels=1, net_depth=4) # This is for patchsize 256x256 pixels
    
    model.to(device)
    model.load_state_dict(torch.load(weights_path))

    pred = np.zeros((patches_loader.shape[0], PATCHSIZE, PATCHSIZE))
    idx = 0
    for img in tqdm(patches_loader):
        img = np.expand_dims(img, 3)
        img = img.transpose((3, 2, 0, 1))
        img = torch.from_numpy(img)
        img = img.to(device, dtype=torch.float)

        with torch.no_grad():
                pred_mask = model(img)
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask = pred_mask.float()
                pred_mask = pred_mask.cpu()
        
        pred[idx,:,:] = pred_mask
        idx = idx + 1
    
    # Reconstruct patch using the dimensions from the original image.
    print(f'Size of predictions: {pred.shape}, data type: {pred.dtype}')
    pred = np.reshape(pred, (patches_size[0], patches_size[1], patches_size[3],patches_size[4]))
    print(f'Reconstructing for unpatchify: {pred.shape}, data type: {pred.dtype}')
    drone_mask = unpatchify(pred, (drone_img_padding_size[0], drone_img_padding_size[1]) )
    print(f'Size of original drone image: {drone_img_padding_size}. Size of mask: {drone_mask.shape, drone_mask.dtype}')

    return drone_mask[0:drone_img_size[0], 0: drone_img_size[1]]

=======
import gc
import cv2
import math
import time
import glob
>>>>>>> 42951068e7560a02fbd4c24434948ed3b02e4130
import torch
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
from src.lib.models.unet import Unet
import torchvision.transforms.functional as F

from PIL import Image
from tqdm import tqdm
from patchify import patchify, unpatchify

def generate_patches(drone_img):
    # Pad the entire image to avoid losing border information using patchify
    row_pad = int(np.ceil(drone_img.shape[0]/PATCHSIZE)*PATCHSIZE - drone_img.shape[0])
    col_pad = int(np.ceil(drone_img.shape[1]/PATCHSIZE)*PATCHSIZE - drone_img.shape[1])
    drone_img_padding = np.pad(drone_img, ((0, row_pad), (0, col_pad), (0,0)), 'constant')

    # All patches in a single array of size (img.shape[0]/PATCHSIZE, img.shape[1]/PATCHSIZE, 1, PATCHSIZE, PATCHSIZE, 3)
    patches = patchify(drone_img_padding, (PATCHSIZE, PATCHSIZE, 3), step=PATCHSIZE) 
    patches_loader = np.reshape(patches, (patches.shape[0]*patches.shape[1]*patches.shape[2],patches.shape[3],patches.shape[4],patches.shape[5]))
    print(f'Size of patches (patchify): {patches.shape}, data type: {patches.dtype}')
    print(f'Size of patch loader: {patches_loader.shape}, data type: {patches_loader.dtype}')

    return patches_loader, patches.shape, drone_img_padding.shape


def generate_probability_map_per_class(patches_loader, patches_size, drone_img_padding_size, drone_img_size, PATCHSIZE, weights_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if PATCHSIZE == 512:
        model = Unet(inchannels=3, outchannels=1, net_depth=5) # This is for patchsize 512x512 pixels
    if PATCHSIZE == 256:
        model = Unet(inchannels=3, outchannels=1, net_depth=4) # This is for patchsize 256x256 pixels
    
    model.to(device)
    model.load_state_dict(torch.load(weights_path))

    pred = np.zeros((patches_loader.shape[0], PATCHSIZE, PATCHSIZE))
    idx = 0
    for img in tqdm(patches_loader):
        img = np.expand_dims(img, 3)
        img = img.transpose((3, 2, 0, 1))
        img = torch.from_numpy(img)
        img = img.to(device, dtype=torch.float)

        with torch.no_grad():
                pred_mask = model(img)
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask = pred_mask.float()
                pred_mask = pred_mask.cpu()
        
        pred[idx,:,:] = pred_mask
        idx = idx + 1
    
    # Reconstruct patch using the dimensions from the original image.
    print(f'Size of predictions: {pred.shape}, data type: {pred.dtype}')
    pred = np.reshape(pred, (patches_size[0], patches_size[1], patches_size[3],patches_size[4]))
    print(f'Reconstructing for unpatchify: {pred.shape}, data type: {pred.dtype}')
    drone_mask = unpatchify(pred, (drone_img_padding_size[0], drone_img_padding_size[1]) )
    print(f'Size of original drone image: {drone_img_padding_size}. Size of mask: {drone_mask.shape, drone_mask.dtype}')

    return drone_mask[0:drone_img_size[0], 0: drone_img_size[1]]

<<<<<<< HEAD
=======
# ------------ #
# Main program #
# ------------ #
>>>>>>> 07140eaaa10e283042a6faa4b7862e3d6610012f
>>>>>>> 42951068e7560a02fbd4c24434948ed3b02e4130

if __name__ =='__main__':

    ''' Output: Predicted mask for each class'''

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
<<<<<<< HEAD
    parser.add_argument("--patch_size", type=int, default=256)
=======
<<<<<<< HEAD
    parser.add_argument("--patch_size", type=int, default=256)
=======
    parser.add_argument("--patch_size", type=int, default=128)
>>>>>>> 07140eaaa10e283042a6faa4b7862e3d6610012f
>>>>>>> 42951068e7560a02fbd4c24434948ed3b02e4130

    args = parser.parse_args()
    
    IMAGEPATH = args.image_path
    OUTPUTDIR = args.output_dir
    PATCHSIZE = args.patch_size
<<<<<<< HEAD
    VEGETATEDUNET= "./results/results-exp-000/vegetated_512x512/exp_CGIAR_vegetated/exp_CGIAR_vegetated_test_00_cv_00_patchSize_512.pt"
    NONVEGETATEDUNET="./results/results-exp-000/non_vegetated_512x512/exp_CGIAR_non_vegetated/exp_CGIAR_non_vegetated_test_00_cv_01_patchSize_512.pt"
    ROADSUNET= "./results/results-exp-000/roads_512x512/exp_CGIAR_roads/exp_CGIAR_roads_test_00_cv_02_patchSize_512.pt"
    TILLAGEUNET= "./results/results-exp-000/tillage_512x512/exp_CGIAR_tillage/exp_CGIAR_tillage_test_00_cv_02_patchSize_512.pt"
    CROPSUNET= "./results/results-exp-000/crops_512x512/exp_CGIAR_crops/exp_CGIAR_crops_test_00_cv_00_patchSize_512.pt"   
    BUILDSINGUNET = "./results/results-exp-000/buildings_512x512/exp_CGIAR_buildings/exp_CGIAR_buildings_test_00_cv_02_patchSize_512.pt"
=======
<<<<<<< HEAD
    VEGETATEDUNET= "./results/vegetated_256x256/exp_CGIAR_vegetated/exp_CGIAR_vegetated_test_00_cv_01_patchSize_256.pt"
    NONVEGETATEDUNET="./results/non_vegetated_256x256/exp_CGIAR_non_vegetated/exp_CGIAR_non_vegetated_test_00_cv_00_patchSize_256.pt"
    ROADSUNET= "./results/roads_256x256/exp_CGIAR_roads/exp_CGIAR_roads_test_00_cv_01_patchSize_256.pt"
    TILLAGEUNET= "./results/tillage_256x256/exp_CGIAR_tillage/exp_CGIAR_tillage_test_00_cv_02_patchSize_256.pt"
    CROPSUNET= "./results/crops_256x256/exp_CGIAR_crops/exp_CGIAR_crops_test_00_cv_00_patchSize_256.pt"   
    BUILDSINGUNET = "./results/buildings_256x256/exp_CGIAR_buildings/exp_CGIAR_buildings_test_00_cv_00_patchSize_256.pt"
=======
    VEGETATEDUNET= "../results/results-exp-000/vegetated_512x512/exp_CGIAR_vegetated/exp_CGIAR_vegetated_test_00_cv_00_patchSize_512.pt"
    NONVEGETATEDUNET="../results/results-exp-000/non_vegetated_512x512/exp_CGIAR_non_vegetated/exp_CGIAR_non_vegetated_test_00_cv_01_patchSize_512.pt"
    ROADSUNET= "../results/results-exp-000/roads_512x512/exp_CGIAR_roads/exp_CGIAR_roads_test_00_cv_02_patchSize_512.pt"
    TILLAGEUNET= "../results/results-exp-000/tillage_512x512/exp_CGIAR_tillage/exp_CGIAR_tillage_test_00_cv_02_patchSize_512.pt"
    CROPSUNET= "../results/results-exp-000/crops_512x512/exp_CGIAR_crops/exp_CGIAR_crops_test_00_cv_00_patchSize_512.pt"   
    BUILDSINGUNET = "../results/results-exp-000/buildings_512x512/exp_CGIAR_buildings/exp_CGIAR_buildings_test_00_cv_02_patchSize_512.pt"
>>>>>>> 07140eaaa10e283042a6faa4b7862e3d6610012f
>>>>>>> 42951068e7560a02fbd4c24434948ed3b02e4130

    try:
        img = tiff.imread(IMAGEPATH) # reading the image as it is going to be used by all threads
        img = img[:,:,:3]
    except:
        print(f'Error in image: {IMAGEPATH}')

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 42951068e7560a02fbd4c24434948ed3b02e4130
    # Patches only need to be computed once, as it is the same image to be processed by multiple networks.
    patches_loader, patches_size, drone_img_padding_size = generate_patches(img)

    crops_pred = generate_probability_map_per_class(patches_loader, patches_size, drone_img_padding_size, img.shape, PATCHSIZE, CROPSUNET)                     # Crops
    roads_pred = generate_probability_map_per_class(patches_loader, patches_size, drone_img_padding_size, img.shape, PATCHSIZE, ROADSUNET)                     # Roads
    tillage_pred = generate_probability_map_per_class(patches_loader, patches_size, drone_img_padding_size, img.shape, PATCHSIZE, TILLAGEUNET)                 # Tillage
    buildings_pred = generate_probability_map_per_class(patches_loader, patches_size, drone_img_padding_size, img.shape, PATCHSIZE, BUILDSINGUNET)             # Buildings
    vegetated_pred = generate_probability_map_per_class(patches_loader, patches_size, drone_img_padding_size, img.shape, PATCHSIZE, VEGETATEDUNET)             # Vegetated water
    non_vegetated_pred = generate_probability_map_per_class(patches_loader, patches_size, drone_img_padding_size, img.shape, PATCHSIZE, NONVEGETATEDUNET)      # Non vegetated water

    output_classes_probability_map = np.zeros((img.shape[0], img.shape[1], 6), dtype=np.float)
    # outputPrediction = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    output_classes_probability_map[:,:,0] = crops_pred
    output_classes_probability_map[:,:,1] = roads_pred
    output_classes_probability_map[:,:,2] = tillage_pred
    output_classes_probability_map[:,:,3] = buildings_pred
    output_classes_probability_map[:,:,4] = vegetated_pred
    output_classes_probability_map[:,:,5] = non_vegetated_pred

    print(output_classes_probability_map.shape, output_classes_probability_map.dtype)

    '''
    How the classes are set up:
    crops -> 1
    roads -> 2
    tillage -> 3
    building -> 4
    vegetated -> 5
    non-vegetated -> 6
    '''

<<<<<<< HEAD
=======
    cropdic= {"probabilities": crops_pred, "label":"crop_map"}
    savemat(f'output_probabilities_crops_{PATCHSIZE}x{PATCHSIZE}.png', cropdic)

    roadsdic= {"probabilities": roads_pred, "label":"roads_map"}
    savemat(f'output_probabilities_roads_{PATCHSIZE}x{PATCHSIZE}.png', roadsdic)

    tillagedic= {"probabilities": tillage_pred, "label":"tillage_map"}
    savemat(f'output_probabilities_tillage_{PATCHSIZE}x{PATCHSIZE}.png', tillagedic)

    buildingsdic= {"probabilities": buildings_pred, "label":"buildings_map"}
    savemat(f'output_probabilities_buildings_{PATCHSIZE}x{PATCHSIZE}.png', buildingsdic)

    vegetateddic= {"probabilities": vegetated_pred, "label":"vegetated_map"}
    savemat(f'output_probabilities_vegetated_{PATCHSIZE}x{PATCHSIZE}.png', vegetateddic)

    nonvegetateddic= {"probabilities": non_vegetated_pred, "label":"nonvegetated_map"}
    savemat(f'output_probabilities_nonvegetated_{PATCHSIZE}x{PATCHSIZE}.png', nonvegetateddic)

>>>>>>> 42951068e7560a02fbd4c24434948ed3b02e4130
    max_index = np.argmax(output_classes_probability_map, axis=2)
    output_mask_th_0p5 = (((np.max(output_classes_probability_map, axis=2) > 0.5) * 1.0) * (max_index + 1)).astype('uint8')
    output_mask_th_0p6 = (((np.max(output_classes_probability_map, axis=2) > 0.6) * 1.0) * (max_index + 1)).astype('uint8')
    output_mask_th_0p7 = (((np.max(output_classes_probability_map, axis=2) > 0.7) * 1.0) * (max_index + 1)).astype('uint8')
    output_mask_th_0p8 = (((np.max(output_classes_probability_map, axis=2) > 0.8) * 1.0) * (max_index + 1)).astype('uint8')
    output_mask_th_0p9 = (((np.max(output_classes_probability_map, axis=2) > 0.9) * 1.0) * (max_index + 1)).astype('uint8')
    
    foreground_0p5 = Image.fromarray(output_mask_th_0p5).point(lambda i: int(i/6 * 255))
    foreground_0p5.save(os.path.join(OUTPUTDIR, f'output_drone_mask_grayscale_th_0p5_{PATCHSIZE}x{PATCHSIZE}.png'))
<<<<<<< HEAD
=======

    foreground_0p6 = Image.fromarray(output_mask_th_0p6).point(lambda i: int(i/6 * 255))
    foreground_0p6.save(os.path.join(OUTPUTDIR, f'output_drone_mask_grayscale_th_0p6_{PATCHSIZE}x{PATCHSIZE}.png'))

    foreground_0p7 = Image.fromarray(output_mask_th_0p7).point(lambda i: int(i/6 * 255))
    foreground_0p7.save(os.path.join(OUTPUTDIR, f'output_drone_mask_grayscale_th_0p7_{PATCHSIZE}x{PATCHSIZE}.png'))

    foreground_0p8 = Image.fromarray(output_mask_th_0p8).point(lambda i: int(i/6 * 255))
    foreground_0p8.save(os.path.join(OUTPUTDIR, f'output_drone_mask_grayscale_th_0p8_{PATCHSIZE}x{PATCHSIZE}.png'))

    foreground_0p9 = Image.fromarray(output_mask_th_0p9).point(lambda i: int(i/6 * 255))
    foreground_0p9.save(os.path.join(OUTPUTDIR, f'output_drone_mask_grayscale_th_0p9_{PATCHSIZE}x{PATCHSIZE}.png'))

    background_0p5 = Image.fromarray(img)
    background_0p5.paste(foreground_0p5, (0,0), foreground_0p5)
    background_0p5.save(os.path.join(OUTPUTDIR, f'output_drone_mask_all_classes_th_0p5_{PATCHSIZE}x{PATCHSIZE}.png'))

    background_0p6 = Image.fromarray(img)
    background_0p6.paste(foreground_0p6, (0,0), foreground_0p6)
    background_0p6.save(os.path.join(OUTPUTDIR, f'output_drone_mask_all_classes_th_0p6_{PATCHSIZE}x{PATCHSIZE}.png'))

    background_0p7 = Image.fromarray(img)
    background_0p7.paste(foreground_0p7, (0,0), foreground_0p7)
    background_0p7.save(os.path.join(OUTPUTDIR, f'output_drone_mask_all_classes_th_0p7_{PATCHSIZE}x{PATCHSIZE}.png'))

    background_0p8 = Image.fromarray(img)
    background_0p8.paste(foreground_0p8, (0,0), foreground_0p8)
    background_0p8.save(os.path.join(OUTPUTDIR, f'output_drone_mask_all_classes_th_0p8_{PATCHSIZE}x{PATCHSIZE}.png'))

    background_0p9 = Image.fromarray(img)
    background_0p9.paste(foreground_0p9, (0,0), foreground_0p9)
    background_0p9.save(os.path.join(OUTPUTDIR, f'output_drone_mask_all_classes_th_0p9_{PATCHSIZE}x{PATCHSIZE}.png'))
=======
>>>>>>> 42951068e7560a02fbd4c24434948ed3b02e4130

    foreground_0p6 = Image.fromarray(output_mask_th_0p6).point(lambda i: int(i/6 * 255))
    foreground_0p6.save(os.path.join(OUTPUTDIR, f'output_drone_mask_grayscale_th_0p6_{PATCHSIZE}x{PATCHSIZE}.png'))

    foreground_0p7 = Image.fromarray(output_mask_th_0p7).point(lambda i: int(i/6 * 255))
    foreground_0p7.save(os.path.join(OUTPUTDIR, f'output_drone_mask_grayscale_th_0p7_{PATCHSIZE}x{PATCHSIZE}.png'))

    foreground_0p8 = Image.fromarray(output_mask_th_0p8).point(lambda i: int(i/6 * 255))
    foreground_0p8.save(os.path.join(OUTPUTDIR, f'output_drone_mask_grayscale_th_0p8_{PATCHSIZE}x{PATCHSIZE}.png'))

    foreground_0p9 = Image.fromarray(output_mask_th_0p9).point(lambda i: int(i/6 * 255))
    foreground_0p9.save(os.path.join(OUTPUTDIR, f'output_drone_mask_grayscale_th_0p9_{PATCHSIZE}x{PATCHSIZE}.png'))

    background_0p5 = Image.fromarray(img)
    background_0p5.paste(foreground_0p5, (0,0), foreground_0p5)
    background_0p5.save(os.path.join(OUTPUTDIR, f'output_drone_mask_all_classes_th_0p5_{PATCHSIZE}x{PATCHSIZE}.png'))

    background_0p6 = Image.fromarray(img)
    background_0p6.paste(foreground_0p6, (0,0), foreground_0p6)
    background_0p6.save(os.path.join(OUTPUTDIR, f'output_drone_mask_all_classes_th_0p6_{PATCHSIZE}x{PATCHSIZE}.png'))

    background_0p7 = Image.fromarray(img)
    background_0p7.paste(foreground_0p7, (0,0), foreground_0p7)
    background_0p7.save(os.path.join(OUTPUTDIR, f'output_drone_mask_all_classes_th_0p7_{PATCHSIZE}x{PATCHSIZE}.png'))

    background_0p8 = Image.fromarray(img)
    background_0p8.paste(foreground_0p8, (0,0), foreground_0p8)
    background_0p8.save(os.path.join(OUTPUTDIR, f'output_drone_mask_all_classes_th_0p8_{PATCHSIZE}x{PATCHSIZE}.png'))

<<<<<<< HEAD
    background_0p9 = Image.fromarray(img)
    background_0p9.paste(foreground_0p9, (0,0), foreground_0p9)
    background_0p9.save(os.path.join(OUTPUTDIR, f'output_drone_mask_all_classes_th_0p9_{PATCHSIZE}x{PATCHSIZE}.png'))
=======
>>>>>>> 07140eaaa10e283042a6faa4b7862e3d6610012f
>>>>>>> 42951068e7560a02fbd4c24434948ed3b02e4130


 
