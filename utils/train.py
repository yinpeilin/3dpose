from torch.nn.parallel import DataParallel
import sys
sys.path.append("./models")
from C3dmodel import Residual3DCNN18
from dataset import VideoDataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam,SGD
import json 
from tqdm import tqdm
import os
import numpy as np
def saveModel(model,model_name):
    torch.save(model.state_dict(), model_name)

def getDataSet(FineDivingPath):
    trainDataSet = VideoDataset(FineDivingPath, train=True)
    testDataSet = VideoDataset(FineDivingPath, train=False)
    train_dataloader = DataLoader(trainDataSet, batch_size = 1, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(testDataSet, batch_size = 1, shuffle=False, num_workers=3)

    return train_dataloader, test_dataloader
def evaluate(model,test_dataloader):
    
    model.eval()

    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    with torch.no_grad():
        for datas,labels in test_dataloader:
            datas = datas.cuda()
            labels = labels.cuda()
            # run the model on the test set to predict labels
            outputs1, outputs2, outputs3 = model(datas)
            # the label with the highest energy will be our prediction
            
            if outputs1.shape[0] == 1:
                
                loss1 += torch.abs((outputs1[0] - labels[0][0])) 
                loss2 += torch.abs((outputs2[0] - labels[0][1])) 
                loss3 += torch.abs((outputs3[:] - labels[0][2:])) 
            else:
                loss1 += torch.abs((outputs1[:] - labels[:][0])) 
                loss2 += torch.abs((outputs2[:] - labels[:][1])) 
                loss3 += torch.abs((outputs3[:] - labels[:][2:])) 
    
    return torch.sum(loss1),torch.sum(loss2),torch.sum(loss3)

def train(FineDivingPath):
    train_dataloader, test_dataloader = getDataSet(FineDivingPath)
    FineDivingDataPaths = json.load(open(FineDivingPath, 'r'))
    max_epoch = FineDivingDataPaths['basic']['max_epoch']
    
    
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("The model will be running on", device, "device")
    
    # model.to(device)
    
    # TODO: 多卡训练
    device_ids=[0, 1]
    model = Residual3DCNN18(102)
    model.load_state_dict(torch.load("models/model_75.pth"))
    model = DataParallel(model, device_ids).module
    optimizer = Adam(model.parameters(),lr=0.0001,weight_decay=0.00001)
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids).module
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10*len(
    #     train_dataloader), 50*len(
    #     train_dataloader), 100*len(
    #     train_dataloader), 130*len(train_dataloader)], gamma=0.2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ 50*len(
        train_dataloader), 130*len(
        train_dataloader)], gamma=0.1)
    loss_fn1 = nn.L1Loss(reduction="sum")
    loss_fn2 = nn.MSELoss(reduction="sum")
    # 返回gpu名字，设备索引默认从0开始
    print(torch.cuda.get_device_name(1))  # index 是索引, 默认从 0 开始

    
    test_loss_list = []
    train_loss_list = []
    running_loss = 0.0
    for epoch in range(0, max_epoch+1):
       
        # train
        # optimizer = Adam(model.parameters(),lr=0.01*(max_epoch+1-epoch)/max_epoch, weight_decay=0.00001)
        # optimizer = nn.DataParallel(optimizer, device_ids=device_ids).module
        train_dataloader, test_dataloader = getDataSet(FineDivingPath)
        model.train()
        model = model.cuda()
        
        with tqdm(total=len(train_dataloader)) as t:
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_loss3 = 0.0
            for i,(datas,labels) in enumerate(train_dataloader, 0):
            # datas = torch.tensor(datas)
            # labels = torch.tensor(labels)
                
                
                datas = datas.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            

                optimizer.zero_grad()
            # predict classes using images from the training set
                outputs1, outputs2, outputs3 = model(datas)
            # compute the loss based on model output and real labels
                # print(labels.shape)
                # print(outputs1.shape)
                # print(outputs2.shape)
                # print(outputs3.shape)
                if outputs1.shape[0] == 1:
                    loss1 = loss_fn1(outputs1[0], labels[0][0])
                    loss2 = 10*loss_fn1(outputs2[0], labels[0][1])
                    loss3 = 0.001*loss_fn2(outputs3[:], labels[0][2:])
                else:
                    loss1 = loss_fn1(outputs1[:], labels[:][0])
                    loss2 = 10*loss_fn1(outputs2[:], labels[:][1])
                    loss3 = 0.001*loss_fn2(outputs3[:], labels[:][2:])
                
                loss = loss1 + loss2 + loss3 

            # backpropagate the loss
                loss.backward()
            # adjust parameters based on the calculated gradients
                optimizer.step()
                scheduler.step()
                # Let's print statistics for every 1,000 images
                running_loss1 += loss1.item()     # extract the loss value
                running_loss2 += loss2.item()
                running_loss3 += loss3.item()
                t.set_description('Epoch %i' % (epoch+1))
                t.set_postfix(target1 = labels[0][0].item(),target2=labels[0][1].item(),predit1 = outputs1.item(), predit2 = outputs2.item(),loss1 = loss1.item(), loss2 = loss2.item(), loss3 = loss3.item())
                t.update(1)

                
                
                if i % 100 == 99:    
                    # print every 1000 (twice per epoch) 
                    print('[%d, %5d] loss1: %.5f loss2: %.5f loss3: %.5f learning rate: %f' %
                          (epoch + 1, i + 1, running_loss1/100, running_loss2/100, running_loss3/100, scheduler.get_last_lr()[0]))
                
                    train_loss_list.append([running_loss1,running_loss2,running_loss3])

                    # zero the loss
                    running_loss1 = 0.0
                    running_loss2 = 0.0
                    running_loss3 = 0.0

        
        

        # TODO: epoch记得改回去
        if epoch %  5 == 0:
            loss1,loss2,loss3 = evaluate(model, test_dataloader)
            loss = [loss1.to('cpu'), loss2.to('cpu'), loss3.to('cpu')]
            print('[%d] test_loss1: %.5f test_loss2: %.5f test_loss3: %.5f' %
                      (epoch + 1, loss[0], loss[1], loss[2]))
            
            test_loss_list.append(loss)
            
            if loss == min(test_loss_list):
                saveModel(model, './models/model_min_' + str(epoch) + '.pth')
            else:
                saveModel(model, './models/model_' + str(epoch) + '.pth')
            
            
            np.savetxt('./models/train_loss.txt', train_loss_list)
            np.savetxt('./models/test_loss.txt', test_loss_list)


    
if __name__ == '__main__':
    FineDivingPath = "FineDiving.json"
    train(FineDivingPath=FineDivingPath)


    