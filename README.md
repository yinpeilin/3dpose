
# 介绍

- [三个精度的模型下载](https://pan.baidu.com/s/1Y-THpfuIqLnZTvaCKRvgKg?pwd=pdap)（提取码为pdap）:
  1. model_0.pth 是训练轮数较少的模型，测试集误差比训练集误差更小，训练集误差约为百分之八，训练集误差约为百分之十
  2. model_1.pth 是训练轮数中等的模型，比较稳定，测试集和训练集误差相近，在百分之七左右
  3. model_2.pth 是训练轮数最多的模型，在跳水分数上总体误差最小，训练集最高误差不超过百分之五，但不稳定，测试集上表现较差，测试集误差约为百分之十
- 主要使用的为一个具有18层的三维残差卷积神经网络，具体的模型架构在models/C3dmodel.py中
# 运行

- 安装必要库（requirement.txt）,还要安装配置[ffmpeg](https://github.com/FutaAlice/ffmpeg-static-libs/releases)
- 进入到main.py同级目录下, 直接运行main.py



# 数据集
该模型使用的数据来源为

["FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality Assessment"](https://finediving.ivg-research.xyz/)
# 待完成事项

* [X] 增加数据噪声
* [X] 增加动作识别模块
* [X] 增加难度识别模块
* [X] test修改
