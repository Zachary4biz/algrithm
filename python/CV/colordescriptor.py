# -*-coding:utf-8-*-

'''
本文件是用HSV颜色空间下的颜色直方图来量化描述图像的一个方法文件
预期在HSV颜色空间中使用3D颜色直方图，
8个bin用于色相通道、12个bin用于饱和度通道、3个bin用于明度通道
特征向量共有8x12x3=288个
每个bin保存的是
意味着最后每幅图像都会用288个浮点数构成的列表量化表示
'''

'''
Q1: 直方图归一化，均衡化？
Q2:8个bin用于色相通道、12个bin用于饱和度通道、3个bin用于明度通道，
总共的特征向量有8 × 12 × 3=288还是8+12+3=23个？以及为什么这样分配？
'''
import numpy as np
import cv2


# 定义描述图像特征的类
class ColorDescriptor:
    def __init__(self, bins):
        # bins表示引用这个类是就要给出的参数，本例中始终为ColorDescriptor((8,12,3))
        self.bins = bins

    def histogram(self, image, mask):
        # 其参数image为要计算的图像，mask为要图像中要计算的区域
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        # 上为通过cv2提供的方法计算mask区域的直方图使用构造器中的bin数目作为参数
        hist = cv2.normalize(hist, hist).flatten()
        # 上为对直方图归一化，即bin所占的比例而不是个数来表示直方图，
        # 归一化之后，当图像尺寸变化是由于是bin的比例，所以还能识别
        return hist

    def describe(self, image):
        # 把图像转换到HSV颜色空间并初始化
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # features用来量化图像
        features = []
        # 获得维度即图像的长款h,w，并计算图像的中心(cX, cY)
        (h, w) = image.shape[:2]
        #		print image.shape    #test
        #		print (h,w)          #test
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        # 把图像分割成四个区域segment左（右）上，左（右）下，分别匹配各个区域的特征值
        # 不然只能获得整幅图中多少比例是蓝多少比例是绿这种，不精确
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
        # np就是numpy，import时as为了np
        # 下列1，2与原教程相反
        # 1建立一个黑色的和图像大小相同的蒙版，命名为ellipMask，之后会在其上画椭圆
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        # 2再用一个椭圆表示图像的中心区域，椭圆的长短轴分别为图像长款的75%
        (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
        # 建立椭圆，以下是ellipse函数说明
        # ellipse(要绘制的位置,(中心坐标),(长短轴),(整体旋转角度)，(画椭圆时起始)，(画椭圆结束角),(颜色，255为白),(边框粗细，-1表示填充) )
        cv2.ellipse(ellipMask, (int(cX), int(cY)), (int(axesX), int(axesY)), 0., 0., 360., (255, 255, 255), -1)
        # 在那四块区域里时，进行循环
        for (startX, endX, startY, endY) in segments:
            # 给每个角的蒙版分配内存
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            # print(cornerMask)  # test
            # 每个角都画一个白色矩形，相当于把图像分割为四个矩形，
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            # 下面从之前分好的四个矩形cornerMask中减去中心的椭圆区域ellipMask
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            # 接下来从图像的cornerMaks中获取颜色直方图，并放到特征向量里
            hist = self.histogram(image, cornerMask)
            features.extend(hist)
        # 从椭圆区域ellipMaks区域获取颜色直方图并放到特征向量
        hist = self.histogram(image, ellipMask)
        features.extend(hist)
        return features
# 上面的蒙版同PS的蒙版，即白色为有，蒙版为白色时才计算直方图
# 直观效果可以看文件中的“蒙版部分示意图.gif”
