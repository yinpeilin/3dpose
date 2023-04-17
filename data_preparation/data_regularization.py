import copy
import os
import pickle
import json
import glob
import cv2
import numpy as np
# load数据
def loadImages(ImageList,addnoise = False):
    Images = []
    for ImagePath in ImageList:
        image = cv2.imread(ImagePath)
        if addnoise:
            mean = 0

            sigma = 5
            #根据均值和标准差生成符合高斯分布的噪声
            gauss = sigma*np.random.randn(image.shape[0],image.shape[1],3)
            
            
            #给图片添加高斯噪声
            image = image + gauss
            
            #设置图片添加高斯噪声之后的像素值的范围
            image = np.clip(image,a_min=0,a_max=255)
            image = cv2.resize(image, (270, 180))
            
            
            image = np.array(image, dtype=np.uint8).copy()
            
            

        Images.append(image)
        
    return Images

def imageProcess(Images, maxLength,primitiveResults):
    
    
    newResults = copy.deepcopy(primitiveResults)
    
    if len(Images) < maxLength:
        newResults = np.concatenate(
            (newResults, [newResults[-1]]*(maxLength-len(Images))), axis=0)
        Images += ([Images[-1]]*(maxLength-len(Images)))
        
        
    # 长了的数据进一步抽帧
    elif len(Images) > maxLength:
        newImages = []
        newResults = np.array([],dtype=np.int64)
        ratio = len(Images)/(maxLength+1)
        while len(newImages) < maxLength:
            temp = int(ratio*len(newImages))
            newImages.append(Images[temp])
            newResults = np.concatenate((newResults,[primitiveResults[temp]]),axis=0)
        
        Images = newImages 
    return (Images, newResults)
    


    


# 保存在DivingData文件中
def saveImages(Images,ImageDir):
    if os.path.exists(ImageDir):
        pass
    else:
        os.makedirs(ImageDir)
    
    for imageCount in range(1,len(Images)+1):
        imagePath = ImageDir + '/'+str(imageCount)+'.jpg'
        Images[imageCount-1] = cv2.resize(Images[imageCount-1],
                                          (270, 150), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(imagePath,Images[imageCount-1])
        np.save(ImageDir + '/'+str(imageCount)+'.npy',np.array(Images[imageCount-1],dtype=np.float64)/255)
        

    


def processData(FineDivingPath:str, newImageDir:str, newResultPath:str):
    
    
    FineDivingDataPaths = json.load(open(FineDivingPath, 'r'))
    
    
    
    max_length = FineDivingDataPaths['basic']['max_frame_length']
    
    DataSetPaths = FineDivingDataPaths['dataSet']
    
    DataRoot = DataSetPaths["data_root"]
    SubActionTypeResultPath = DataSetPaths['sub_action_result']
    SubActionTypeResult = pickle.load(open(SubActionTypeResultPath, 'rb'))
    
    
    newSubActionTypeResult = copy.deepcopy(SubActionTypeResult)
   
    count = 0
    for key, value in SubActionTypeResult.items():
        
        
        ImagesRelativeDir = '/'+key[0] + '/' + str(key[1])
        ImageDir = DataRoot+ImagesRelativeDir
        image_list = glob.glob(ImageDir+"/*.jpg")
        
        images = loadImages(image_list)
        images, newSubActionTypeResult[key]['frames_labels'] = imageProcess(images, max_length, value['frames_labels'])
        
        new_image_dir = newImageDir + ImagesRelativeDir
        saveImages(images, new_image_dir)

        
        count+=1
        if count%3 == 0:
            print('第{}份照片处理完成！,照片数为{},frame长度为{}'.format(count, len(images), newSubActionTypeResult[key]['frames_labels'].shape[0]))

        
    
    with open(newResultPath, 'wb') as fp:
        pickle.dump(newSubActionTypeResult, fp)

    print('\n照片全部处理完成！')



if __name__ == '__main__':
    FineDivingPath = "FineDiving.json"
    newImageDir = "./datasets/DivingData"
    newResultPath = "./Annotations/PROCESSED_FineDiving_fine-grained_annotation.pkl"
    processData(FineDivingPath,newImageDir,newResultPath)
    
    