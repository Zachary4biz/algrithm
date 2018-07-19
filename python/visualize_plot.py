# encoding=utf-8
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np

# 设置画布大小
fig_ori = plt.figure(figsize=(12,5))

def demo_add_subplot(fig):
    # ---- add_subplot
    # 添加第一个子图 表示 按布局是:2行2列 取第1个图的位置给ax1
    ax1 = fig.add_subplot(221)
    # 添加第二个子图 表示 按布局是:2行2列 取第2个图的位置给ax2
    ax2 = fig.add_subplot(222)
    # 添加第二个子图 表示 按布局是:2行3列 取第4个图(即左下)的位置给ax3
    ax3 = fig.add_subplot(234)
    plt.show()



