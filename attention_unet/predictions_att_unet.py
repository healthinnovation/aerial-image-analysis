from matplotlib.image import imsave
import torch
import numpy as np
from unet import AttU_Net_heat_map
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image, ImageStat
import os
from os import listdir

# Class to evaluate
classFolder = "vegetated/"

#Paths 
csvFileTest = "../data/experiment_vegetated_000/test_00.csv"
pathOutputPrediction= "../results/att_unet/prediction_vegetated_256/"

folderImagePathRoot = '../../dataset_256x256/'
fileNetworkWeights="../../best-models-att-unet-256x256/heat_experiments_vegetated/test_00_attention_unet_EP_60_ES_10_BS_8_LR_0.01_RS_2022/ckpts/FOLD-3/best_model.pth"

def single_dice_coeff(input_bn_mask, true_bn_mask):
    """single_dice_coeff : function that returns the dice coeff for one pair 
    of mask and ground truth mask"""

    # The eps value is used for numerical stability
    eps = 0.0001

    # Computing intersection and union masks
    inter_mask = np.dot(input_bn_mask.flatten(), true_bn_mask.flatten())
    union_mask = np.sum(input_bn_mask) + np.sum(true_bn_mask) + eps

    # Computing the Dice coefficient
    return (2 * inter_mask.astype(float) + eps) / union_mask.astype(float)


def example_plot(ax, fontsize=12, hide_labels=False):
        ax.plot([1, 2])

        ax.locator_params(nbins=3)
        if hide_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xlabel('x-label', fontsize=fontsize)
            ax.set_ylabel('y-label', fontsize=fontsize)
            ax.set_title('Title', fontsize=fontsize)

    

def showPredicted(img, groundTruth, pred, dicePred, fileName):
    
    ''' Function that plots the image, gt, predicted image, the value of dice and save it in png'''
    #print(type(groundTruth), type(np.array(pred.convert('1'))))
    #print(groundTruth, np.array(pred.convert('1')))

    fig, (axs1, axs2, axs3) = plt.subplots(1, 3, constrained_layout=True,figsize=(7,3))
    #fig.suptitle("Dice %f"% dicePred)
    axs1.set_xticklabels([])
    axs1.set_yticklabels([])
    axs1.axis('off') 
    axs1.set_title('Image', fontsize=10)
    axs1.imshow(img)

    axs2.set_xticklabels([])
    axs2.set_yticklabels([])
    axs2.axis('off') 
    axs2.set_title('Ground truth', fontsize=10)    
    axs2.imshow(groundTruth) 
    
    axs3.set_xticklabels([])
    axs3.set_yticklabels([])
    axs3.axis('off') 
    axs3.set_title('Prediction (Dice: %.2f@0.65)'% dicePred, fontsize=10)
    axs3.imshow(pred.convert('1')) 

    fig.savefig(fileName)
    plt.close()

if __name__ == '__main__':

    # Model and weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttU_Net_heat_map(img_ch=3, output_ch=1)
    model.to(device)
    model.load_state_dict(torch.load(fileNetworkWeights))

    #print("Model loaded")
    
    # Make a folder for the predicted images
    predictionFolder = os.path.join(pathOutputPrediction, classFolder)
    os.makedirs(predictionFolder, exist_ok=True)
    #print(f'\nCreating folder {predictionFolder}')

    #print("Folder created")

    df = pd.read_csv(csvFileTest)
    listFolderTest = df.folder.to_list()

    for folderImageTest in listFolderTest:

        folderImagePath = os.path.join(folderImagePathRoot, folderImageTest, classFolder, 'patches/')
        folderMaskPath = os.path.join(folderImagePathRoot, folderImageTest, classFolder, 'masks/')

        print(folderImagePath)
        onlyfiles = [f for f in listdir(folderImagePath) if os.path.isfile(os.path.join(folderImagePath, f))]
        onlyfiles = sorted(onlyfiles)
        
        numMasks = len(onlyfiles) # one for the image 
        print("numMasks:", numMasks- 1)

        predictionFolderImg = os.path.join(predictionFolder, folderImageTest)
        os.makedirs(predictionFolderImg, exist_ok=True)
        #print(f'\nCreating folder {predictionFolderImg}')


        for i in range(1, numMasks):

            filenamePath = os.path.join(folderImagePath, onlyfiles[i])
            #print(f'\nPredicting image Path {filenamePath}')
            filename = os.path.basename(filenamePath)
            #print(f'\nPredicting image {filename}')

            imgSave = Image.open(filenamePath)
            img = np.array(imgSave)

            #print("Shape img:", img.shape)

            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img,3)
            img = img.transpose((3, 0, 1, 2))

            img = torch.from_numpy(img)

            img = img.to(device, dtype=torch.float)

            with torch.no_grad():
                pred_mask, _ = model(img)
                pred = torch.sigmoid(pred_mask)
                pred = (pred > 0.65).float()
                pred = pred.cpu()
        
            predSave = Image.fromarray(np.uint8(np.squeeze(pred))*255)
            
            #print("Pred type: ", type(pred))
            #print("Pred shape: ", pred.shape)

            #print(f'\n Calculating dice:')

            maskNamePath = os.path.join(folderMaskPath, filename.replace('patch', 'mask')) 
            
            #print(f'\n Calculating dice with Ground truth  {maskNamePath}')      
            
            maskSave = Image.open(maskNamePath).convert('1')
            mask = np.array(maskSave)
            #print(mask*1.0)
            #print(np.array(np.squeeze(pred)))
            #print("Shape mask:", mask.shape)

        
            diceCoef = single_dice_coeff(mask, np.array(np.squeeze(pred)))
            
            #diceCoef = 0.5
            predNamePath =  os.path.join(predictionFolderImg, filename.replace('.png', '_pred.png'))       
            #print(f'Prediction name:  {predNamePath}')

            #print(predNamePath)
            showPredicted(imgSave, maskSave, predSave, diceCoef, predNamePath)
