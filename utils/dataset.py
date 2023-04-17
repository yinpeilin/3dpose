import sys
sys.path.append("./data_preparation")
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import json
import glob
import numpy as np
from data_regularization import loadImages
import cv2
import random
import time

class VideoDataset(Dataset):
    def __init__(self, JSONpath, train = True):
        FineDivingDataPaths = json.load(open(JSONpath, 'r'))
        DataSetPaths = FineDivingDataPaths['dataSet']
        
        
        if train == True:
            self.data_names = pickle.load(open(DataSetPaths['train_split'], 'rb'))
        else:
            self.data_names = pickle.load(open(DataSetPaths['test_split'], 'rb'))
            
        SubActionTypeResultPath = DataSetPaths['process_action_result']
        SubActionTypeResult = pickle.load(open(SubActionTypeResultPath, 'rb'))
        self.results_dict = SubActionTypeResult
        self.length = len(self.data_names)
        self.data_root = DataSetPaths["new_data_root"]
        
        
        # print(self.data_names)
        # print(self.results_dict)
    def __getitem__(self, index):
        
        index = index % self.length
        
        
        name_dir = self.data_names[index]
        ImagesRelativeDir = '/' + name_dir[0] + '/' + str(name_dir[1])
        ImageDir = self.data_root+ImagesRelativeDir
        image_list = glob.glob(ImageDir+"/*.jpg")
        
        # images = []
        # image_numpy_list = glob.glob(ImageDir+"/*.npy")
        
        images = loadImages(image_list, addnoise=True)
        
        # for image_numpy_path in image_numpy_list:
        #     images.append(np.load(image_numpy_path))
        R_list = []
        G_list = []
        B_list = []
        
        for image in images:
            
            R,G,B = cv2.split(image)
            R_list.append(R)
            G_list.append(G)
            B_list.append(B)# list of arrays
            # images[i] = np.array([R,G,B],dtype=np.uint8)
        # 归一化
    
        images_matrix = np.array([R_list, G_list, B_list],dtype = np.float32)/255.0
        random.seed(time.time())
        
        if random.random()>0.5:
            images_matrix = np.flip(images_matrix,axis=3)
            
        # print(images_matrix.shape)

        
        # score 0~1
        score = np.array(self.results_dict[name_dir]['dive_score'],dtype = np.float32)
        
        # difficult 
        difficulty = (np.array(self.results_dict[name_dir]['difficulty'],dtype = np.float32))
        
        #subtype 
        subtype = np.array(self.results_dict[name_dir]['frames_labels'],dtype = np.float32)
        label = np.array([score, difficulty])
        label = np.append(label, subtype)
        
        
        # print(label)
        # TODO： label添加
        return images_matrix.copy(),label.copy()
    
    def __len__(self):
        return self.length


if __name__ == '__main__':
    FineDivingPath = "FineDiving.json"
    trainDataSet = VideoDataset(FineDivingPath,train=True)
    testDataSet = VideoDataset(FineDivingPath,train=False)

    train_dataloader = DataLoader(trainDataSet, batch_size=3, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(testDataSet, batch_size=3, shuffle=False, num_workers=6)

    max_difficulty = 0.0
    min_difficulty = 10.0
    for images,labels in train_dataloader:
        # print(images.shape)
        # print(labels.shape)
        

        print(labels.shape)
    # print(max_difficulty)
    # print(min_difficulty)