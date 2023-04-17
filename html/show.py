import sys
sys.path.append("./")

from tools.VideoPoseGet import Paint3d, GetPoseDataFromVideo
from tools.video2images import video2frames, FramesChange
import numpy as np
from models.C3dmodel import Residual3DCNN18
import torch
import streamlit as st
import os
import json
import pickle
from sklearn.cluster import KMeans
class modelShow:
    def __init__(self):
        st.title('VideoPose')
        self.video_path = st.text_input('video_path', './example/test.mp4')
        self.model_path = st.text_input('model_path', './models/model_0.pth')
        self.device = st.selectbox(
            'Which device do you want the model to run on?',
            ['cpu']+['cuda'])
        self.actionNum = int(st.text_input('actionNum', '4'))
        self.isStart = st.button("start", key=None)
        self.model = Residual3DCNN18(102)
        self.model.load_state_dict(torch.load(os.path.abspath(self.model_path)))
        
        self.subtype = pickle.load(open(json.load(open("FineDiving.json", 'r'))["dataSet"]["sub_action"], 'rb'))
        self.result = []

        if self.device == 'cpu':
            self.model.to(torch.device('cpu'))
        else:
            self.model.to(torch.device('cuda'))
        self.frames = None

        if self.isStart:
            # try:
            self.video_abspath = os.path.abspath(self.video_path)
            self.ShowVideo(self.video_abspath)
            
            
            
            
            
            # except:
            # st.error("请正确输入视频地址")


            self.GetResult()
            self.ShowResult()

        pass

    def GetResult(self):
        with st.spinner(text='In progress...'):
            self.frames = video2frames(self.video_abspath)
            
            # 获得pose
            self.frame_poses, self.frame_num = GetPoseDataFromVideo(self.frames)
            
            # 获得model导入格式
            self.frames = FramesChange(self.frames)
            with torch.no_grad():
                self.frames = torch.tensor(self.frames)
                self.frames = self.frames.to(torch.device(self.device))
                results = self.model(self.frames)

                for result in results:
                    result = result.to('cpu')
                    self.result.append(np.array(result))
            
    def ShowVideo(self, video_abspath):
        try:
            st.video(video_abspath)
        except:
            st.error("the video path is invalid.")

    def ShowBar(self):
        pass

    def ShowResult(self):
        
        # 计算难度系数，作为平均值的平均值，并且它也是最后一个
        self.result[2] = np.array(self.result[2], dtype=np.int32)
        # 将一系列整数压缩为四个整数的序列，确保所有四个整数都是正数，并确保它们之间的损失尽可能小，使用这个短向量来表示原始向量
        sequence = []
        for i in range(self.result[2].shape[0]):
            if self.result[2][i] > 0:
                sequence.append( [i,self.result[2][i]])
                
        sequence = np.array(sequence, dtype = np.float32) 
        kmeans = KMeans(n_clusters=self.actionNum, random_state=0, max_iter = 1000)
        compressed = kmeans.fit_predict(sequence)  # 找到四个整数的分组，并将它们映射到四个角色中，
        # random_state=0 是保
        action_type_all = [0] * self.actionNum
        action_count = [0] * self.actionNum
        for i in range(sequence.shape[0]):
            action_count[compressed[i]] += 1
            action_type_all[compressed[i]] += sequence[i][1]
        action_type = [int(action_type_all[i]/action_count[i]) for i in range(self.actionNum)]
        
        type_result = []
        for i in range(1,len(compressed)):
            if compressed[i-1]!= compressed[i]:
                type_result.append(action_type[compressed[i-1]])
            elif i == len(compressed)-1:
                type_result.append(action_type[compressed[i]])
        print(sequence)
        print(type_result)
        st.markdown("## 1. 得分为" + "**{:.3f}**".format(self.result[0][0]))
        st.markdown("## 2. 难度系数为" + "**{:.3f}**".format(self.result[1][0]))
        st.markdown("## 3. 识别到的动作有：")
        for i in range(len(type_result)):
            st.markdown(" - "+"**{}**".format(self.subtype[type_result[i]]))
        
        
        Paint3d(self.frame_poses, self.frame_num).GetAnimation()
        
        st.markdown("## 4. 动作识别效果：")
        st.video("./example/poseGet.mp4")
        
        
        
        st.balloons()
        st.success("Success Process!")


if __name__ == "__main__":
    model_show = modelShow()
