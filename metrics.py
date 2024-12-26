from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import lpips
import os
import pandas as pd
from torchvision.transforms import transforms
import warnings
import matplotlib.pyplot as plt
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

warnings.filterwarnings("ignore")


def calc_ssim(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("L")
    try:
        img2 = Image.open(img2_path).convert("L")
    except:
        img2 = Image.open(img2_path.replace(".png", ".jpg", ".JPG")).convert("L")
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255)
    return ssim_score


def calc_psnr(img1_path, img2_path):
    img1 = Image.open(img1_path)
    try:
        img2 = Image.open(img2_path)
    except:
        img2 = Image.open(img2_path.replace(".png", ".jpg", ".JPG"))
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score


class util_of_lpips:
    def __init__(self, net=None, use_gpu=False):
        ## Initializing the model
        net = "vgg"  # ['squeeze', 'alex', 'vgg']
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1_path, img2_path):
        # Load images
        img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
        try:
            img1 = lpips.im2tensor(lpips.load_image(img2_path))
        except:
            img1 = lpips.im2tensor(lpips.load_image(img2_path.replace(".png", ".jpg",".JPG")))
        if img1.shape != img0.shape:
            resize_transform = transforms.Resize(img0.shape[2:])
            img1 = resize_transform(img1)
        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = self.loss_fn.forward(img0, img1).item()
        return dist01





import cv2
import numpy as np


def compute_local_contrast(image):
    mean = cv2.boxFilter(image, -1, (3, 3))
    mean_sq = cv2.boxFilter(image**2, -1, (3, 3))
    local_contrast = np.sqrt(mean_sq - mean**2)
    return local_contrast


def compute_niqe_score(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    except:
        image = cv2.imread(
            image_path.replace(".png", ".jpg",".JPG"), cv2.IMREAD_GRAYSCALE
        ).astype(np.float32)
    local_contrast = compute_local_contrast(image)
    mean_contrast = np.mean(local_contrast)
    std_contrast = np.std(local_contrast)
    niqe_score = std_contrast / mean_contrast
    return niqe_score


if __name__ == "__main__":
    mode = 'lol'
    #mode = 'lol_blur'
    
    lol_blur_gt ="/data0/model_dataset/LOL_blur/eval/high_sharp_scaled"
    lol_gt =  "/data0/model_dataset/deepretinnex/LOLdataset/eval15/high"
    

    lol_gt = lol_gt if mode == 'lol' else lol_blur_gt
    
    


    a = util_of_lpips(use_gpu=True)

    #path = './lol'
    path = '/home/SCI/EXP/Train-20240829-042757/image_epochs'
    #path = '/home/SCI/EXP/Train-20240829-145132/image_epochs'
    lpipss, ssims, psnrs, niqe = 0, 0, 0, 0
    for q, single_image in enumerate(os.listdir(lol_gt)):
    
        img2_path = os.path.join(path, single_image)
        img2_path = img2_path.split('.')[0]+'_99.'+img2_path.split('.')[-1]
        lpipss += a.calc_lpips(
            os.path.join(lol_gt, single_image), img2_path
        )
        ssims += calc_ssim(
            os.path.join(lol_gt, single_image), img2_path
        )
        psnrs += calc_psnr(
            os.path.join(lol_gt, single_image), img2_path
        )
        niqe += compute_niqe_score(img2_path)
    
    lpipss /= q + 1
    ssims /= q + 1
    psnrs /= q + 1
    niqe /= q + 1
    print("lpips:", lpipss)
    print("ssim:", ssims)
    print("psnr:", psnrs)
    print("niqe:", niqe)

    # print('lpips:',a.calc_lpips("./pics/DUAL_result_wx.jpg", "./pics/wx.jpg"))
    # print('ssim:',calc_ssim("./pics/DUAL_result_wx.jpg", "./pics/wx.jpg"))
    # print('psnr:', calc_psnr("./pics/DUAL_result_wx.jpg", "./pics/wx.jpg"))

    # image_path = "./pics/wx.jpg"
    # niqe_score = compute_niqe_score(image_path)
    # print("NIQE score of the image: ", niqe_score)

