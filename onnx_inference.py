import numpy as np
from PIL import Image 
import onnxruntime as ort
def preprocess_input(x):
    x /= 255.
    x -= np.array([0.485, 0.456, 0.406])
    x /= np.array([0.229, 0.224, 0.225])
    return x

def read_img_lable(file='/home/low_light_enhancement/classification-pytorch_1/cls_test.txt'):
    with open(file,'r')as f:
        a = f.readlines()
        a = [i.replace('\n','').split(';') for i in a]
        a = np.array(a,dtype=object)
    return a[:,1],a[:,0]


def predict(image_path = 'path_to_image.jpg', ort_session=None):
    image = Image.open(image_path).resize((224,224)).convert('RGB')
    image = preprocess_input(np.array(image,dtype=np.float32))
    image = np.transpose(image,(2,0,1))
    # 添加批次维度
    image = np.expand_dims(image, axis=0)

    # 进行推理
    outputs = ort_session.run(['output'], {'input': image})

    # 获取分类结果
    preds = np.array(outputs).squeeze()
    predicted_class = np.argmax(preds)
    return predicted_class

if __name__ == '__main__':
    onnx_model_path = 'utils/resnet50.onnx'
    ort_session = ort.InferenceSession(onnx_model_path)
    imgs,labels = read_img_lable()
    print(labels[-3:])
    print([predict(imgs[len(imgs)-3+i],ort_session) for i in range(3)])