import argparse
import os
import matplotlib.pyplot as plt
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument(
    "--lowlight_images_path",
    type=str,
    default="/data0/model_dataset/",
)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--grad_clip_norm", type=float, default=0.1)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--train_batch_size", type=int, default=8) #8
parser.add_argument("--val_batch_size", type=int, default=4) #4
parser.add_argument("--num_workers", type=int, default=4)  #除了llformer，其他时候为4
parser.add_argument("--display_iter", type=int, default=100)
parser.add_argument("--snapshot_iter", type=int, default=10)
parser.add_argument("--snapshots_folder", type=str, default="snapshots/")
parser.add_argument("--load_pretrain", type=bool, default=True)
parser.add_argument("--pretrain_dir", type=str, default="snapshots/pretrain.pth")
parser.add_argument("--eval_epoch", type=int, default=1)
parser.add_argument("--save_epoch",type=int,default=50)
parser.add_argument("--he", type=lambda x: (str(x).lower() == 'true'), default=True,help='是否使用辅助损失')
parser.add_argument("--mode", type=str, default="leq",choices=['llformer','lime','eq','leq','origin'])
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--domain_adaption", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--dataset",type=str,default='dice',choices=['zero','dice'])
config = parser.parse_args()
if not os.path.exists(config.snapshots_folder):
    os.mkdir(config.snapshots_folder)
print(f'Using the {config.gpu} gpu!')
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import csv
import sys
import time
from sklearn.metrics import accuracy_score 
# import dataloader
from dataloader import LowlightDataset
import model
import Myloss
import numpy as np
from torchvision import transforms
from metrics import util_of_lpips, calc_ssim, calc_psnr, compute_niqe_score
import clip

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class High_Frenquency:
    def __init__(self):
        h,w = 224,224
        lpf = torch.zeros((h,w))
        R = (h+w)//8  #或其他
        for x in range(w):
            for y in range(h):
                if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
                    lpf[y,x] = 1
        hpf = 1-lpf
        self.hpf, lpf = hpf, lpf
    def get_fft(self,x):
        f = torch.fft.fftn(x,dim=(2,3))
        f = torch.roll(f,(224//2,224//2),dims=(2,3))
        f = f * self.hpf
        return torch.abs(torch.fft.ifftn(f,dim=(2,3)))

class Eval:
    def __init__(
        self,
        dataloader,
        label_path="/data0/model_dataset/deepretinnex/LOLdataset/eval15/low/",
        onnx_model_path = 'utils/resnet50.onnx',
        ex_path = '/home/low_light_enhancement/classification-pytorch_1/cls_test.txt'
    ):
        self.data = dataloader
        self.label_path = label_path
        self.a = util_of_lpips(use_gpu=True)
        self.metrics = [['lpips','ssim','psnr','niqe','acc']]
        self.max_ssim = 0
        self.max_psnr = 0
        self.max_acc = 0

          # 替换为你的 ONNX 模型路径
        self.ort_session = ort.InferenceSession(onnx_model_path,providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.exdark_img,self.exdark_lable = self.read_img_lable(ex_path,to_tensor=True)

    def eval_exdark(self,model,imgs,labels):
        preds = []
        with torch.no_grad():
            for img in imgs:
                _,enhanced,_ = model(img)
                pred_lable = self.predict(enhanced.cpu().numpy(),self.ort_session)
                preds.append(pred_lable)
        return accuracy_score(labels,preds)
    def eval(self, model):
        model.eval()
        lpips, ssim, psnr, niqe, acc = 0, 0, 0, 0, 0
        total_images = len(self.data)  # 记录处理的图像数量

        for q, data in enumerate(self.data):
            lowlight = data["input"].cuda()
            _, enhanced, _ = model(lowlight)

            # 保存增强的图像
            enhanced_image_path = "./temp/" + os.path.basename(data["path"][0])
            torchvision.utils.save_image(enhanced, enhanced_image_path)

            # 获取正常光照标签图像的路径
            # 注意：这里假设 data["path"][0] 返回的相对路径是以 'low' 目录开始的。
            # 您可能需要调整为只取文件名或根据实际路径调整。
            gt_image_path = os.path.join('/data0/model_dataset/deepretinnex/LOLdataset/eval15/high', os.path.basename(data["path"][0]))
            #gt_image_path = os.path.join('/data0/model_dataset/LOL_blur/eval/high_sharp_scaled', os.path.basename(data["path"][0]))

            # 计算 LPIPS、SSIM、PSNR，使用增强图像与正常光照图像进行比较
            lpips_value = self.a.calc_lpips(gt_image_path, enhanced_image_path)
            ssim_value = calc_ssim(gt_image_path, enhanced_image_path)
            psnr_value = calc_psnr(gt_image_path, enhanced_image_path)

            # 计算 NIQE，仅对增强后的图像
            niqe_value = compute_niqe_score(enhanced_image_path)

            # 累加每个指标的值
            lpips += lpips_value
            ssim += ssim_value
            psnr += psnr_value
            niqe += niqe_value

        # 计算 ExDark 数据集上的准确率（如果需要）
        print('Calculating Accuracy on ExDark:')
        acc = self.eval_exdark(model, self.exdark_img, self.exdark_lable)
        print(acc)

        # 将平均计算结果保存
        average_metrics = [
            lpips / total_images,
            ssim / total_images,
            psnr / total_images,
            niqe / total_images,
            acc
        ]
        self.metrics.append(average_metrics)

        return_result = [False, False, False]

        # 更新最大值并判断是否需要返回结果
        if ssim / total_images > self.max_ssim:
            self.max_ssim = ssim / total_images
            return_result[0] = True
        if psnr / total_images > self.max_psnr:
            self.max_psnr = psnr / total_images
            return_result[1] = True
        if acc > self.max_acc:
            self.max_acc = acc
            return_result[2] = True

        return return_result

    def to_csv(self, file_path):
        # 将指标写入 CSV 文件
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['LPIPS', 'SSIM', 'PSNR', 'NIQE', 'Acc'])  # 表头
            for row in self.metrics:
                writer.writerow(row)




    def read_img_lable(self,file='/home/low_light_enhancement/classification-pytorch_1/cls_test.txt',to_tensor=False):
        with open(file,'r')as f:
            a = f.readlines()
            a = [i.replace('\n','').split(';') for i in a]
            a = np.array(a,dtype=object)
        if to_tensor:
            imgs = []
            for i in a[:,1]:
                imgs.append(ToTensor()(Image.open(i).resize((224,224)).convert('RGB')).cuda().unsqueeze(0))
            return imgs,np.array(a[:,0],dtype=int).reshape(-1)
        return a[:,1],a[:,0]
    def predict(self,image, ort_session=None):
        image = self.preprocess_input(image)
        # 进行推理
        outputs = ort_session.run(['output'], {'input': image})

        # 获取分类结果
        preds = np.array(outputs).squeeze()
        predicted_class = np.argmax(preds)
        return predicted_class
    
    def preprocess_input(self,x):
        x = np.transpose(x[0],(1,2,0))
        # x /= 255.
        x -= np.array([0.485, 0.456, 0.406])
        x /= np.array([0.229, 0.224, 0.225])
        x = np.transpose(x,(2,0,1))
        x = np.expand_dims(x,0)
        return x
    def to_csv(self, path="1.csv"):
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.metrics)


def train(config):

    if not os.path.exists(os.path.join(config.snapshots_folder,config.dataset)):
        os.mkdir(os.path.join(config.snapshots_folder,config.dataset))
    if not os.path.exists(os.path.join(config.snapshots_folder,config.dataset,config.mode)):
        os.mkdir(os.path.join(config.snapshots_folder,config.dataset,config.mode))

    if config.domain_adaption:
        d_loss_all = []
        g_loss_all = []
        netD = model.Discriminator().cuda()
        optimizerD = torch.optim.Adam(
            netD.parameters(), lr=config.lr*0.1, weight_decay=config.weight_decay
        )
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=10, eta_min=config.lr*1e-2)
        res, _ = clip.load("RN50", device='cuda')
        res.visual.requires_grad_(False)
        res.transformer.requires_grad_(False)
        res.visual.float()
        preprocess = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            # torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        # preprocess = torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        # Loss_cls = nn.BCELoss()
        L_cl = Myloss.MultiLabelContrastiveLoss()
        get_h = High_Frenquency()

    if config.he:
        MSE = torch.nn.MSELoss(reduction='mean')

    if torch.cuda.is_available():
        DCE_net = model.enhance_net_nopool().cuda()
    else:
        DCE_net = model.enhance_net_nopool()

    DCE_net.apply(weights_init)
    if config.load_pretrain == True:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))

    train_dataset = LowlightDataset(config.lowlight_images_path,he=config.he,mode=config.mode,domain=config.domain_adaption,dataset=config.dataset,cache=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_dataset = LowlightDataset(
      "/data0/model_dataset/deepretinnex/LOLdataset/eval15/low/", eval=True
      )
    # test_dataset = LowlightDataset(
    #    "/data0/model_dataset/LOL_blur/eval/low_blur/", eval=True
    #   )
    der = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()

    L_exp = Myloss.L_exp(16, 0.8)
    L_TV = Myloss.L_TV()

        
    optimizer = torch.optim.Adam(
        DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=config.lr*1e-2)



    paired = None
    Eval_class = Eval(test_loader)

    for epoch in range(config.num_epochs):
        d_step = 0
        DCE_net.train()
        if config.domain_adaption:
            temp_g_loss,temp_d_loss = 0,0
        for iteration, data in enumerate(train_loader):
            d_step += 1
            if config.he:
                img_lowlight = data["input"].cuda()
                paired = data["paired"].cuda()
                if config.domain_adaption:
                    img_domain = data['domain'].cuda()
            else:
                img_lowlight = data["input"].cuda()
                if config.domain_adaption:
                    img_domain = data['domain'].cuda()


            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
            # true_label = torch.full((enhanced_image.shape[0],), 1, dtype=torch.float, device=enhanced_image.device)
            # false_label = torch.full((enhanced_image.shape[0],), 0, dtype=torch.float, device=enhanced_image.device)
            # if config.domain_adaption and d_step % 5 == 0:
            #     optimizerD.zero_grad()
            #     real_feature = res.encode_image(preprocess(img_domain)).detach()
            #     fake_feature = res.encode_image(preprocess(enhanced_image)).detach()
            #     fake_feature = netD(fake_feature).squeeze()
            #     real_feature = netD(real_feature).squeeze()
            #     # loss_D = 1 - torch.mean(real_feature) + torch.mean(fake_feature) # -  10*torch.mean(torch.abs(real_feature-fake_feature))
            #     loss_D = Loss_cls(real_feature,true_label) + Loss_cls(fake_feature,false_label)
            #     temp_d_loss += loss_D.item()/len(train_loader)
            #     loss_D.backward()
            #     optimizerD.step()


            Loss_TV = 200 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(enhanced_image))
            
            # best_loss
            loss =  loss_col + loss_exp + loss_spa + Loss_TV

            if config.he:
                loss += 10*MSE(enhanced_image, paired)
            if config.domain_adaption:
                # real_feature = res.encode_image(preprocess(img_domain)).detach()
                # fake_feature = res.encode_image(preprocess(enhanced_image))#.detach()
                real_feature = res.encode_image(get_h.get_fft(img_domain.cpu()).cuda()).detach()
                fake_feature = res.encode_image(get_h.get_fft(enhanced_image.cpu()).cuda())
                r_length,f_length = real_feature.shape[0],fake_feature.shape[0]
                domain_feature = torch.cat([fake_feature,real_feature],dim=0)
                domain_target = torch.full((domain_feature.shape[0],domain_feature.shape[0]), 1, dtype=torch.float, device=enhanced_image.device)
                domain_target[:f_length,:f_length] = 0
                torch.diagonal(domain_target).fill_(1)
                ignore_mask = torch.full((domain_feature.shape[0],domain_feature.shape[0]), 1, dtype=torch.float, device=enhanced_image.device)
                ignore_mask[r_length:,r_length:] = 0
                loss_gen = L_cl(domain_feature,domain_target,ignore_mask)
                temp_g_loss += loss_gen.item()/len(train_loader)
                loss += loss_gen

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()


                
            if ((iteration + 1) % config.display_iter) == 0:
                print("Epoch:",epoch+1,"Loss at iteration", iteration + 1, ":", loss.item())

        scheduler.step()
        if config.domain_adaption:
            schedulerD.step()
        if (1 + epoch) % config.eval_epoch == 0:
            is_max_ssim,is_max_psnr,is_max_acc = Eval_class.eval(DCE_net)
            if is_max_ssim:
                torch.save(
                DCE_net.state_dict(),
                os.path.join(config.snapshots_folder,config.dataset,config.mode,str(config.domain_adaption)+ "best_ssim.pth")
                )
            if is_max_psnr:
                torch.save(
                DCE_net.state_dict(),
                os.path.join(config.snapshots_folder,config.dataset,config.mode,str(config.domain_adaption)+ "best_psnr.pth")
                )
            if is_max_acc:
                torch.save(
                DCE_net.state_dict(),
                os.path.join(config.snapshots_folder,config.dataset,config.mode,str(config.domain_adaption)+ "best_acc.pth")
                )
        
        if (1 + epoch) % config.save_epoch == 0:
            torch.save(
                DCE_net.state_dict(),
                os.path.join(config.snapshots_folder,config.dataset,config.mode,str(config.domain_adaption)+ "_epoch" + str(epoch) + ".pth")
                )
        if config.domain_adaption:
            d_loss_all.append(temp_d_loss)
            g_loss_all.append(temp_g_loss)
            plt.figure()
            # plt.plot(d_loss_all,label='d_loss')
            plt.plot(g_loss_all,label='g_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('adaption_loss.jpg')
    Eval_class.to_csv("./csv/%s_%s_%s.csv"%(config.mode,config.dataset,str(config.domain_adaption)))
if __name__ == "__main__":
    train(config)
