# Object Classification with TensorRT using a pretrained EfficientNetB2 CNN on ImageNet.
# Please see References.md in this repository.
# This script will take images from camera and predict the class of object.
# Please refer to the LICENSE file in this repository.

# coding: utf-8

#Import Packages
import cv2 as cv
import numpy as np
from onnx_helper import ONNXClassifierWrapper,convert_onnx_to_engine
import torch
import json
from PIL import Image
from torchvision import transforms

#Set constants
BATCH_SIZE=1
N_CLASSES=1000
PRECISION=np.float32
image_size=224
TRT_PATH='models/efficientnetb2_batch1.trt'

#Load TensorRT Engine
print("Loading TRT Engine")
trt_model=ONNXClassifierWrapper(TRT_PATH,[BATCH_SIZE,N_CLASSES],target_dtype=PRECISION)
print("Loaded TRT Engine!!")

#Load Labels
print("Loading classification labels")
labels_map=json.load(open('labels_map.txt'))
labels_map=[labels_map[str(i)] for i in range(1000)]


# Function for Inferencing
def infer_objects(image):
    img=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    img=Image.fromarray(img)
    #img=Image.open(image)
    image_size=224
    tfms=transforms.Compose([transforms.Resize(image_size),
                         transforms.CenterCrop(image_size),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #Pytorch tensor transform.
    img=tfms(img)
    img=img.unsqueeze(0)
    
    #Convert to numpy.
    trt_input=img.numpy()

    #Convert input to shape required by TensorRT [batch,W,H,Channels] ->[1,224,224,3]
    trt_input=trt_input.transpose((0,3,2, 1))
    
    #Infer
    predictions=trt_model.predict(trt_input)
    
    #Convert numpy to Torch tensor to get the topK predictions.
    predt=torch.from_numpy(predictions)
    preds=torch.topk(predt,k=1).indices.squeeze(0).tolist()

    for idx in preds:
        label=labels_map[idx]
        prob=torch.softmax(predt,dim=1)[0,idx].item()
    return prob,label
    

#Starting the camera. Please modify the cv.VideoCapture(x)
# x is the camera number.

print("Start Classifying...")
capture=cv.VideoCapture(1) #USB camera number 1.
print("Camera started...")

#Loops until Enter Key is pressed on the keyboard
while (True):
    ret, img=capture.read()
    prob,label=infer_objects(img)
    cv.putText(img, label, (int(50), int(30)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv.imshow('Image',img)
    if cv.waitKey(1) == 13: #13 is the Enter Key
        break
print("Releasing Image Window !")        
capture.release()
cv.destroyAllWindows()
print("Exiting !.")
exit()

