import os
import sys

import torch
import torch.utils.data as data
from torchvision.transforms import ToTensor
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from PIL import Image,ImageOps,ImageFile
import glob
import random
import cv2
import threading
from utils.LIME import DUAL
from utils.LLFormer import LLFormerInfer
random.seed(1143)


# 设置PIL处理图像文件截断错误
ImageFile.LOAD_TRUNCATED_IMAGES = True
def get_image_paths(root_folder):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 常见的图像文件扩展名列表

    image_paths = []  # 存储所有图像文件的相对路径

    # 使用 os.walk() 遍历 root_folder 及其子文件夹
    for folder_path, _, file_names in os.walk(root_folder):
        for file_name in file_names:
            # 检查文件扩展名是否为图像扩展名之一
            _, extension = os.path.splitext(file_name)
            if extension.lower() in image_extensions:
                # 构建相对根目录的路径
                relative_path = os.path.relpath(os.path.join(folder_path, file_name), root_folder)
                image_paths.append(relative_path)

    return image_paths

class HE:
    def __init__(self, mode='eq'):
        self.mode = mode
        if self.mode == 'lime':
            self.lime = DUAL()
        elif self.mode == 'llformer':
            self.llformer = LLFormerInfer()

    def histogram_equalization_rgb(self,image):
        # 将图像分离为R、G、B三个通道
        r, g, b = image.split()
        # 对每个通道分别进行直方图均衡化
        r_eq = ImageOps.equalize(r)
        g_eq = ImageOps.equalize(g)
        b_eq = ImageOps.equalize(b)
        # 将均衡化后的通道合并成一个图像
        equalized_image = Image.merge("RGB", (r_eq, g_eq, b_eq))
        return equalized_image

    def clahe_equalization_rgb(self,image, clip_limit=2.0, tile_grid_size=(8, 8)):
        # 将图像分离为R、G、B三个通道
        r, g, b = image.split()
        
        # 将PIL图像转换为NumPy数组
        r = np.array(r)
        g = np.array(g)
        b = np.array(b)
        
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # 对每个通道分别进行局部直方图均衡化
        r_eq = clahe.apply(r)
        g_eq = clahe.apply(g)
        b_eq = clahe.apply(b)
        
        # 将均衡化后的通道合并成一个图像
        equalized_image = Image.merge("RGB", (Image.fromarray(r_eq), Image.fromarray(g_eq), Image.fromarray(b_eq)))
        
        return equalized_image
    
    def __call__(self, img,dataset,name,size=224):
        if os.path.exists('./paired/{0}/{1}/{2}'.format(dataset,self.mode, name)):
            return Image.open('./paired/{0}/{1}/{2}'.format(dataset,self.mode, name))
        if self.mode == 'eq':
            rs = self.histogram_equalization_rgb(img).resize((size,size),Image.LANCZOS)
            rs.save('./paired/{0}/eq/{1}'.format(dataset,name))
            return rs
        elif self.mode == 'lime':
            rs = self.lime(img).resize((size,size),Image.LANCZOS)
            rs.save('./paired/{0}/lime/{1}'.format(dataset,name))
            return rs
        elif self.mode == 'llformer':
            rs = self.llformer(img).resize((size,size),Image.LANCZOS)
            rs.save('./paired/{0}/llformer/{1}'.format(dataset,name))
            return rs
        elif self.mode == 'leq':
            rs = self.clahe_equalization_rgb(img).resize((size,size),Image.LANCZOS)
            rs.save('./paired/{0}/leq/{1}'.format(dataset,name))
            return rs
    


def populate_train_list(lowlight_images_path):

    image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg") + glob.glob(lowlight_images_path + "*.png")

    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list


class LowlightDataset(data.Dataset):

    def __init__(self, lowlight_images_path, eval=False,he=False,mode='eq',domain=False,dataset=None,cache=False):
        if dataset == 'zero' or eval==True:
       
            self.train_list = populate_train_list(lowlight_images_path)
            self.base_data_path = lowlight_images_path
        else:
            self.train_list = get_image_paths('/data0/model_dataset/LOL_blur/train')
            self.train_list = [os.path.join('/data0/model_dataset/LOL_blur/train',i) for i in self.train_list]
            self.base_data_path = '/data0/model_dataset/LOL_blur/train'
        self.size = 224
        self.data_list = self.train_list
        self.cache = cache
        if self.cache==True:
            self.PIL_he = {}
            self.lock = threading.Lock()
            self.data_list_new = []
            # self.PIL_list = []
            # for _ in self.data_list:
            #     self.PIL_list.append(Image.open(_).resize((self.size,self.size),Image.LANCZOS))
            self.PIL_list = []
            with ThreadPoolExecutor() as executor:
                list(tqdm(executor.map(self.process_and_store_image_PIL, self.data_list), total=len(self.data_list), desc="Loading images"))
            print('Total PIL:',len(self.PIL_list))
            self.data_list = self.data_list_new
        print("Total training examples:", len(self.train_list))

        self.eval = eval
        self.he = he
        if self.he:
            self.dataset = dataset
            self.he = HE(mode)
        self.totensor = ToTensor()

        self.domain = domain
        if self.domain:
            self.domain_list = get_image_paths('/data0/model_dataset/cocoval2017/')
            self.domain_list = [os.path.join('/data0/model_dataset/cocoval2017',i) for i in self.domain_list]
            print('Total domain examples:',len(self.domain_list))
            self.domain_length = len(self.domain_list)
            if self.cache==True:
                self.PIL_domain_list = []
                self.domain_list_new = []
                with ThreadPoolExecutor() as executor:
                    list(tqdm(executor.map(self.process_and_store_image_PIL_domain, self.domain_list), total=len(self.domain_list), desc="Loading images"))
                self.domain_list = self.domain_list_new
                # for _ in self.domain_list:
                #     self.PIL_domain_list.append(Image.open(_).resize((self.size,self.size),Image.LANCZOS))
                print('Total domain PIL:',len(self.PIL_domain_list))


    def process_and_store_image_PIL(self, path):
        img = Image.open(path).convert('RGB').resize((self.size, self.size), Image.LANCZOS)
        with self.lock:
            self.PIL_list.append(img)
            self.data_list_new.append(path)
    def process_and_store_image_PIL_domain(self, path):
        img = Image.open(path).convert('RGB').resize((self.size, self.size), Image.LANCZOS)
        with self.lock:
            self.PIL_domain_list.append(img)
            self.domain_list_new.append(path)
    
    def __getitem__(self, index):

        data_lowlight_path = self.data_list[index]
        if self.cache:
            data_lowlight = self.PIL_list[index]
        else:
            data_lowlight = Image.open(data_lowlight_path).convert('RGB')
        if self.he and not self.eval:
            if self.cache:
                gen_paired = self.PIL_he.get(str(index))
                if gen_paired == None:
                    gen_paired = self.he(data_lowlight,self.dataset,'_'.join(data_lowlight_path.replace(self.base_data_path,'').split('/')))
                    self.PIL_he[str(index)] = gen_paired
            else:
                gen_paired = self.he(data_lowlight,self.dataset,'_'.join(data_lowlight_path.replace(self.base_data_path,'').split('/')))
            gen_paired = self.totensor(gen_paired)
        if not self.eval:
            data_lowlight = data_lowlight.resize((self.size, self.size), Image.LANCZOS)
            if self.domain:
                domain_index = random.randint(0, self.domain_length - 1)
                if self.cache:
                    data_domain = self.PIL_domain_list[domain_index]
                else:
                    domain_path = self.domain_list[domain_index]
                    data_domain = Image.open(domain_path).convert('RGB')
                    data_domain = data_domain.resize((self.size, self.size), Image.LANCZOS)
                data_domain = self.totensor(data_domain)

        data_lowlight = self.totensor(data_lowlight)
        # data_lowlight = np.asarray(data_lowlight) / 255.0
        # data_lowlight = torch.from_numpy(data_lowlight).float()
        if self.eval:
            return {"input": data_lowlight, "path": data_lowlight_path}
        # if data_lowlight.shape[0] == 1 or gen_paired.shape[1] == 1 or data_domain.shape[1] == 1:
        # print(data_lowlight.shape, gen_paired.shape, data_domain.shape)
        if self.he:
            if self.domain:
                return {"input": data_lowlight, "paired": gen_paired, "domain": data_domain}
            return {"input": data_lowlight, "paired": gen_paired}
        else:
            if self.domain:
                return {"input": data_lowlight, "domain": data_domain}
            return {"input":data_lowlight}

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train_dataset = LowlightDataset("/data0/model_dataset/train_data/", he=True, mode='llformer')
    for data in train_dataset:

        continue