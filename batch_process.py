


from flir_image_extractor import *
import os
from PIL import Image
from tqdm import tqdm
from batch_utils import *
from joblib import Parallel,delayed


# input folder
input_folder = '/Users/rc/Documents/data_from_imperial/FLIR_100'

# set output folder
folder_dir = setFolderDirs(input_folder)

def imageConvert(img_path):
    # process images
    img_file = img_path.split('/')[-1]
    fie = FlirImageExtractor()
    fie.process_image(img_path)

    # extract images
    img_visual, img_thermal = extractImgs(fie)
    
    # save images
    save_all_images(img_file, folder_dir, img_visual, img_thermal)



# batch process images insde input folder
bacth_img_list =[]
print('Scanning images in the folder')
for img_file in tqdm(os.listdir(input_folder)):
    if img_file.split('.')[-1] == 'jpg':
        bacth_img_list.append(os.path.join(input_folder,img_file))
print('There are {} images need to be processed'.format(len(bacth_img_list)))


# for img_path in tqdm(bacth_img_list): 
#     imageConvert(img_path)

Parallel(n_jobs=-1)(delayed(imageConvert)(img_path) 
											for img_path in tqdm(bacth_img_list))