# get the pose data
import sys
sys.path.append("./")
from tools.video2images import video2frames
import matplotlib.pyplot as plt
import mediapipe as mp 
import cv2
import numpy as np
import matplotlib.animation as animation


def imageReader(IMGFile = None):
    if IMGFile == None:
        print("No input")
        return None
    elif IMGFile != None:
        return cv2.imread(IMGFile)
        
    
def GetPoseDataFromVideo(video = None):
    '''
    return a list of dicts
    '''
    assert video != None
    n = 0 #总的帧数
    data_list = [] #每个时刻的数据
    for frame in video:
        n += 1
        data_list.append(GetPoseDataFromPicture(image = frame))
        print("第{}帧处理完成".format(n),end=', ')
        try:
            len(data_list[-1])
            print("发现有效坐标")
        except Exception as e:
            print("但未发现有效坐标")
            
    return data_list,n
    
def GetPoseDataFromPicture(image = None,imageFile = None):
    '''
    return a dict  
    '''
    p_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5)
    
    if image.all() == None:
        image = imageReader(imageFile)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    BG_COLOR = (192, 192, 192) # gray
    data = {}
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.pose_world_landmarks == None:
        return None
    data = results.pose_world_landmarks.landmark
    #print(results.pose_world_landmarks .landmark)
    return data
    """
    The 33 pose landmarks.
        NOSE = 0
        LEFT_EYE_INNER = 1
        LEFT_EYE = 2
        LEFT_EYE_OUTER = 3
        RIGHT_EYE_INNER = 4
        RIGHT_EYE = 5
        RIGHT_EYE_OUTER = 6
        LEFT_EAR = 7
        RIGHT_EAR = 8
        MOUTH_LEFT = 9
        MOUTH_RIGHT = 10
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_PINKY = 17
        RIGHT_PINKY = 18
        LEFT_INDEX = 19
        RIGHT_INDEX = 20
        LEFT_THUMB = 21
        RIGHT_THUMB = 22
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_HEEL = 29
        RIGHT_HEEL = 30
        LEFT_FOOT_INDEX = 31
        RIGHT_FOOT_INDEX = 32
    """

def PaintLine(pointsData ,ax):
    pair_list = [
        (0, 1),
        (0, 4),
        (1, 3),
        (4, 6),
        (3, 7),
        (6, 8),
        (9, 10),
        (12, 14),
        (14, 16),
        (12, 11),
        (11, 13),
        (13, 15),
        (12, 24),
        (11, 23),
        (24, 23),
        (24, 26),
        (26, 28),
        (23, 25),
        (25, 27),
    ]
    for index1, index2 in pair_list:
        ax.plot([pointsData[index1].x, pointsData[index2].x], [pointsData[index1].z, pointsData[index2].z],
                [-pointsData[index1].y, -pointsData[index2].y], c='r', linewidth=1, marker=".", markeredgecolor='b', markerfacecolor='b')



class Paint3d():
    def __init__(self, datalist, frameNum):
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='3d'))
        self.tmpData = None
        self.data = datalist
        self.frameNum = frameNum
        
    def draw(self):
        self.GetAnimation()
        plt.show()
    def GetAnimation(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.FramePaint, frames=np.arange(0, self.frameNum), interval=80)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='ypl'), bitrate=1800)
        self.ani.save("./example/poseGet.mp4",writer=writer)
    def FramePaint(self,i):
        plt.cla()
        self.ax.set_title('Relative Coordinates')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        
        # ax.plot([datas[i][0].x,datas[i][1].x],[datas[i][0].y,datas[i][1].y],[datas[i][0].z,datas[i][1].z], c = 'r', linewidth=1, marker=".", markeredgecolor='b', markerfacecolor='b')
        try:
            PaintLine(self.data[i], self.ax)
            self.tmpData = self.data[i]

        except:
            PaintLine(self.tmpData, self.ax)
        
if __name__ == '__main__':
    print("hello world")
    data_list, frameNum = GetPoseDataFromVideo(video2frames("./example/test.mp4"))
    Paint3d(data_list, frameNum).draw()