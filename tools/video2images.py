import cv2
import os
import numpy as np
def video2frames(video_path, outPutDirName = None,frame_limit=100):
    times = 0
    
    # 提取视频的频率，每1帧提取一个
	# 如果文件目录不存在则创建目录
    if outPutDirName is not None: 
        if not os.path.exists(outPutDirName):
            os.makedirs(outPutDirName)
        
    # 读取视频帧
    camera = cv2.VideoCapture(video_path)
    
    while True:
        times = times + 1
        res, image = camera.read()
        if not res:
            break
        # 按照设置间隔存储视频帧
            
    ratio = times/frame_limit
    
    
    camera = cv2.VideoCapture(video_path)

    temp1 = 0
    images = []
    temp2 = 0 
    while True:
        temp1 = temp1 + 1
        res, image = camera.read()
        if not res:
            break
        # 按照设置间隔存储视频帧
        if temp1 >= temp2:
            if outPutDirName is not None:
                cv2.imwrite(outPutDirName + '\\' + str(times)+'.jpg', image)
            
            images.append(image)
            temp2 += ratio


    print('图片抽帧结束')
    # 释放摄像头设备
    
    camera.release()
    print(len(images))
    if len(images) > frame_limit:
        images = images[0:frame_limit]
    elif len(images) < frame_limit:
        images = images + [images[-1]] * (frame_limit - len(images))
    
    return images
def FramesChange(frames, image_size_limit = (270, 180)):
    R_list = []
    G_list = []
    B_list = []
    for frame in frames:
        frame = cv2.resize(frame,image_size_limit)
        R,G,B = cv2.split(frame)
        R_list.append(R)
        G_list.append(G)
        B_list.append(B)
    
    frames = np.array([[R_list,G_list,B_list]],dtype = np.float32)/255
    # print(frames.shape)
    # cv2.imshow('123', frames[0][0][0])
    # cv2.waitKey(0)
    return frames