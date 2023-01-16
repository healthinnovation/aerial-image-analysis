import torch
import numpy as np
from lib.models.unet import Unet
from lib import utils
from matplotlib import cm

from PIL import Image, ImageStat
import os
Image.MAX_IMAGE_PIXELS = 377844779

#if __name__ == "__main__":
# drone_img_path = '/home/ubuntu/dataset/MaskBurkinaFaseDL/2018_11_13_1__pissy_mos'
# img_name = 'I_2018_11_13_1__pissy_mos.tif'
# # label_name = 'M_Tillage_2018_11_13_1__pissy_mos.tif'
# label_name = 'M_Roads_2018_11_13_1__pissy_mos.tif'

drone_img_path = '/home/ubuntu/MasksBurkinaFasoDL/2019_06_06_kongtenga_mos'
img_name = 'I_2019_06_06_kongtenga_mos.tif'
label_name = 'M_Tillage_2019_06_06_kongtenga_mos.tif'
# label_name = 'M_Roads_2019_06_06_kongtenga_mos.tif'


img = Image.open(os.path.join(drone_img_path, img_name)).convert('RGB')
label = Image.open(os.path.join(drone_img_path, label_name)).convert('L')

print(img.size, img.size[0], img.size[1])

img_show = img.resize((img.size[0]//8,img.size[1]//8)) # resize four times bigger
label_show = label.resize((img.size[0]//8,img.size[1]//8)) # resize four times bigger
label_show = label_show.point(lambda i: i * 255) 

img_show.save('../results/img.png')
label_show.convert('L').save('../results/labels.png')
# print(np.array(label_show).max())

patchSize = 256
offset = 5

coords = (img.size[0]//offset,img.size[1]//offset,img.size[0]//offset+patchSize,img.size[1]//offset+patchSize)
print(coords)
img = img.crop(coords)
label = label.crop(coords)
label = label.point(lambda i: i * 255) 

img.save('../results/patch.png')
label.convert('L').save('../results/gt.png')

img = np.array(img)
img = img.transpose((2, 0, 1))
img = np.expand_dims(img,3)
img = img.transpose((3, 0, 1, 2))

# print(img.shape)

img = torch.from_numpy(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(inchannels=3, outchannels=1, net_depth=4)
model.to(device)
model.load_state_dict(torch.load('../results/exp_test_01_cv_00_patchSize_256.pt'))

img = img.to(device, dtype=torch.float)
with torch.no_grad():
    pred_mask = model(img)
    pred = torch.sigmoid(pred_mask)
    pred = (pred > 0.5).float()
    pred = pred.cpu()

# print(pred.shape)
pred = Image.fromarray(np.uint8(np.squeeze(pred)*255))
pred.save('../results/prediction.png')