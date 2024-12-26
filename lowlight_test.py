import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def lowlight(DCE_net,image_path,saved_path='./sice_lol'):

    
    data_lowlight = Image.open(image_path).convert('RGB').resize((224,224),Image.LANCZOS)

    data_lowlight = np.asarray(data_lowlight) / 255.0

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    if torch.cuda.is_available():
        data_lowlight = data_lowlight.cuda().unsqueeze(0)
        # DCE_net = model.enhance_net_nopool().cuda()
    else:
        data_lowlight = data_lowlight.unsqueeze(0)
        # DCE_net = model.enhance_net_nopool()

    # DCE_net.load_state_dict(torch.load("snapshots/dice/leq/True_epoch100.pth"))

    _, enhanced_image, _ = DCE_net(data_lowlight)


    # image_path = image_path.replace('test_data','result')
    # result_path = image_path
    # if not os.path.exists(os.path.join("./lol_blur", image_path.split("/")[-2])):
    #     os.mkdir(os.path.join("./lol_blur", image_path.split("/")[-2]))
    # result_path = os.path.join("./lol_blur", "/".join(image_path.split("/")[-2:]))
    

    # if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
    # 	os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

    result_path = os.path.join(f'{saved_path}',os.path.basename(image_path))
    torchvision.utils.save_image(enhanced_image, result_path)
    
    del data_lowlight
    del enhanced_image
    torch.cuda.empty_cache()

    
if __name__ == "__main__":
    # test_images


    if torch.cuda.is_available():
        DCE_net = model.enhance_net_nopool().cuda()
    else:
        DCE_net = model.enhance_net_nopool()

    DCE_net.load_state_dict(torch.load("snapshots/dice/leq/Falsebest_acc.pth"))

    with torch.no_grad():
        # filePath = '/data0/model_dataset/deepretinnex/LOLdataset/eval15/'
        # filePath = "/root/autodl-tmp/datasets/model_dataset/LOL_blur/test/low_blur"
        filePath = '/data0/model_dataset/ExDark2/images'
        
        file_list = os.listdir(filePath)
        for file_name in file_list:
            test_list = glob.glob(filePath + "/" + file_name + "/*")
            for image in tqdm(test_list, desc=f"Processing {file_name} images"):
                saved_path = '/'.join(image.split('/')[:-1]).replace('/images','/zero_leq_best_acc')
                os.makedirs(saved_path,exist_ok=True)
                lowlight(DCE_net,image,saved_path)










