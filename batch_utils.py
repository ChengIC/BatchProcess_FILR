

import os
from PIL import Image
import numpy as np
from matplotlib import cm

def setFolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def setFolderDirs(output_folder):
    thermal_images_folder = os.path.join(output_folder,'batch_process','thermal_images')
    visual_images_folder = os.path.join(output_folder,'batch_process','visual_images')
    overall_images_folder = os.path.join(output_folder,'batch_process','overall_images')

    setFolder(thermal_images_folder)
    setFolder(visual_images_folder)
    setFolder(overall_images_folder)

    folder_dir ={
        'visual':visual_images_folder,
        'thermal':thermal_images_folder,
        'overall':overall_images_folder
    }
    
    return folder_dir

def extractImgs(fie):
    img_visual = Image.fromarray(fie.get_rgb_np())
    img_visual = img_visual.convert('RGB')

    thermal_np = fie.extract_thermal_image()
    thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
    img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))
    img_thermal = img_thermal.convert('RGB')

    return img_visual, img_thermal

def save_all_images(img_file, folder_dir,img_visual, img_thermal):
    th_img_path = os.path.join(folder_dir['thermal'],img_file)
    rgb_img_path = os.path.join(folder_dir['visual'],img_file)
    overall_img_path = os.path.join(folder_dir['overall'],img_file)

    img_thermal.save(th_img_path)
    img_visual.save(rgb_img_path)

    width, height = img_thermal.size
    img_visual = img_visual.resize((width, height))
    get_concat_h(img_visual, img_thermal).save(overall_img_path)
