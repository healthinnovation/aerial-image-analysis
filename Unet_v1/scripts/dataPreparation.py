__author__= "fedra_trujillano"

import os 
from os import listdir
#import gc
import imageio
from PIL import Image
import numpy as np
import csv
import tifffile as tiff


imagePath = '../data_drones' # Path of the folder of the images and masks
patchesFolder = '../data_patches'  # Path to save the patches
rowsCSV = []

def generatePatches(img, mask, imageName, cod, patchSize, percentage):
	
	print ("Creating the DB: ISZ %dx%d ..." %(patchSize, patchSize))

	step= int(round(patchSize))
	area=int(round(patchSize*patchSize*percentage))    
 
	rSize, cSize = img.shape[0], img.shape[1]

	rBegP=0
	rEndP=rBegP+patchSize	
	rMax=rSize-patchSize
	
	cBegP=0
	cEndP=cBegP+patchSize
	cMax=cSize-patchSize

	patchCount = 0 

	print ('step:', step, 'rEndP:', rEndP, 'rMax:', rMax, 'cEndP:', cEndP, 'cMax:', cMax)

	while (cEndP< cMax):
		while(rEndP<rMax):
			patchImg=img[rBegP:rEndP, cBegP:cEndP] 
			patchMask= mask[rBegP:rEndP, cBegP:cEndP]

			if (np.count_nonzero(patchImg[:,:,1]) >= area):
				
				patchCountImg = str(cod)+ '_'+ str(patchCount).zfill(6)+ '.tif'
				patchCountMask = str(cod)+ '_'+ str(patchCount).zfill(6)+ '_mask.tif'

				iPatchIName = os.path.join(patchesFolder, 'images/', patchCountImg)
				mPatchIName = os.path.join(patchesFolder, 'masks/', patchCountMask)

				print(iPatchIName)
				print(mPatchIName)

				imageio.imwrite(iPatchIName, patchImg)
				imageio.imwrite(mPatchIName, patchMask)

				patchCount = patchCount + 1
				rowsCSV.append([patchCountImg, imageName])

			rBegP= rBegP+step
			rEndP= rBegP+patchSize
		cBegP= cBegP+step
		cEndP= cBegP +patchSize 
		rBegP=0
		rEndP=rBegP+patchSize	
   
if __name__ =='__main__':
    

# Open the folder imagePath and iterates over the folders, each folder contains an image and the masks
# Images starts with I and masks with M	
	
	vegetated= 1
	nonVegetated = 2
	buildings = 3
	tillage = 4
	crops = 5
	roads = 6
	
	listFolders= listdir(imagePath)
	patchSize= 512

	
	os.makedirs(os.path.join(patchesFolder,'images/'), exist_ok=True)
	os.makedirs(os.path.join(patchesFolder,'masks/'), exist_ok=True)

	for folder in listFolders:
		folderPath = imagePath + '/' + folder
		print('FolderPath', folderPath)

		onlyfiles = [f for f in listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))]
		onlyfiles = sorted(onlyfiles)
		print('files:', onlyfiles)


		#Image name
		#cod = onlyfiles[0][2:6]
		cod = 'CIV'
		imageFile = os.path.join(folderPath, onlyfiles[0])

		print(cod)

		#Mask name
		

		print('image:', imageFile)
		print('mask:', maskFile)

		image = tiff.imread(imageFile)
		image = image[:,:,0:3]
		print('img', image.shape)
		print('image type:', image.dtype)

	    #Masks 	
	   
	    rSize, cSize = image.shape[0], image.shape[1]	

		mask = np.zeros([rSize, cSize], dtype = int)

		numMasks = len(onlyfiles) - 1 # one for the image 

		for i in range(1, numMasks):

			maskFile = os.path.join(folderPath, onlyfiles[i])
			mask_class = onlyfiles[i][2]

			switch (mask_class):

				case 'N':
					mask 

				break

				case 'B':

				break

				case 'R':

				break

				case 'T':

				break

				case 'V':

				break

				case 'C':

				break


		mask = tiff.imread(maskFile)
		print('mask', mask.shape)

		generatePatches(image, mask, imageFile, cod, patchSize, 0.8)

	csv_fields = ['input_img', 'patch_name']
	csvFile = os.path.join(patchesFolder, 'patch_dataset.csv')
	patch_file_csv = open(csvFile,'w')

	with patch_file_csv:
		write = csv.writer(patch_file_csv)
		write.writerow(csv_fields)
		write.writerows(rowsCSV)

	print(f'{len(rowsCSV)} patches created!')