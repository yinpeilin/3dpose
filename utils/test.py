import sys
sys.path.append("./models")
sys.path.append("./tools")
sys.path.append("./data_preparation")

from video2images import video2frames 
from data_regularization import loadImages
from C3dmodel import C3D

import torch
import numpy as np
import glob
import cv2
def testForVideo(VideoPath,modelPath):
    images = video2frames(VideoPath)
    
    for imageCount in range(1,len(images)+1):
        images[imageCount-1] = cv2.resize(images[imageCount-1],
                                          (270, 150), interpolation=cv2.INTER_CUBIC)
    
    model = C3D()
    
    model.load_state_dict(torch.load(modelPath))
    
    score = model(np.array(images,dtype=np.float64))*100

    print("测试完成,分数为{}".format(score))
    return score
    
def testForImages(ImageDir,modelPath, maxLength = 100):
    
    print("图片测试中！")
    image_list = glob.glob(ImageDir+"/*.jpg")
    images = loadImages(image_list)
    
    if len(Images) < maxLength:
        Images += ([Images[-1]]*(maxLength-len(Images)))
    # 长了的数据进一步抽帧
    elif len(Images) > maxLength:
        newImages = []
        newResults = np.array([],dtype=np.int64)
        ratio = len(Images)/(maxLength+1)
        while len(newImages) < maxLength:
            temp = int(ratio*len(newImages))
            newImages.append(Images[temp])
        Images = newImages 
    
    for imageCount in range(1,len(Images)+1):
        Images[imageCount-1] = cv2.resize(Images[imageCount-1],
                                          (270, 150), interpolation=cv2.INTER_CUBIC)
    
    
    model = C3D()
    model.load_state_dict(torch.load(modelPath))
    
    score = model(np.array(images,dtype=np.float64))*100
    
    print("测试完成,分数为{}".format(score))
    
    return score


if __name__ == '__main__':
    ImageDir = 'datasets/DivingData/01/1'
    testForImages(ImageDir)
    