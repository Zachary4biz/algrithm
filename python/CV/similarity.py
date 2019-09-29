# author: zac
# create-time: 2019-09-28 15:44
# usage: -
import cv2
import numpy as np
from PIL import Image
from functools import reduce

def sim0(p1_inp, p2_inp, compare_method=0):
    # p1 = cv2.cvtColor(cv2.resize(p1_inp,(8,8),interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    # p2 = cv2.cvtColor(cv2.resize(p2_inp,(8,8),interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    p1 = cv2.resize(p1_inp,(8,8),interpolation=cv2.INTER_AREA)
    p2 = cv2.resize(p2_inp,(8,8),interpolation=cv2.INTER_AREA)
    H1 = cv2.calcHist([p1], [0,1,2], None, [32]*3, [0, 256, 0, 256, 0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)
    H2 = cv2.calcHist([p2], [0,1,2], None, [32]*3, [0, 256, 0, 256, 0, 256])
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
    return round(cv2.compareHist(H1, H2, compare_method),4)

def sim(p1_inp, p2_inp, channels=None, compare_method=0):
    if channels is None:
        channels = [0]
    # p1 = cv2.cvtColor(cv2.resize(p1_inp,(8,8),interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    # p2 = cv2.cvtColor(cv2.resize(p2_inp,(8,8),interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    p1 = cv2.resize(p1_inp,(8,8),interpolation=cv2.INTER_AREA)
    p2 = cv2.resize(p2_inp,(8,8),interpolation=cv2.INTER_AREA)
    H1 = cv2.calcHist([p1], channels, None, [8], [0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)
    H2 = cv2.calcHist([p2], channels, None, [8], [0, 256])
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
    # cv2.imshow("h1",drawHist(H1))
    # cv2.imshow("h2",drawHist(H2))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return round(cv2.compareHist(H1, H2, compare_method),4)

def sim2(p1_inp,p2_inp, compare_method=0):
    p1 = cv2.cvtColor(cv2.resize(p1_inp,(8,8),interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    p2 = cv2.cvtColor(cv2.resize(p2_inp,(8,8),interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    H1 = cv2.calcHist([p1], [0], None, [256], [0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)
    H2 = cv2.calcHist([p2], [0], None, [256], [0, 256])
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
    return round(cv2.compareHist(H1, H2, compare_method),4)

def sim_hash(p1_inp, p2_inp):
    def phash(img):
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
        avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
        return reduce(
            lambda x, y: x | (y[1] << y[0]),
            enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())),
            0
        )

    # 比较两个hash指纹的汉明距离
    # hamming(phash(img1),phash(img2))
    def hamming(h1, h2):
        h, d = 0, h1 ^ h2
        while d:
            h += 1
            d &= d - 1
        return h

    return hamming(phash(p1_inp), phash(p2_inp))

def drawHist(hist):
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([hist.shape[0], hist.shape[0], 3], np.uint8)
    hpt = int(0.9 * hist.shape[0])
    for idx,h in enumerate(hist):
        intensity = int(h * hpt / maxVal)
        cv2.line(histImg, (idx, 256), (idx, 256 - intensity), [255, 255, 255])
    return histImg



fname = "f2"
print(fname)
p1_ = "/Users/zac/Downloads/{}.jpg".format(fname)
p2_ = "/Users/zac/Downloads/{}_.jpg".format(fname)

p1_inp = cv2.imread(p1_)
p2_inp = cv2.imread(p2_)
cv2.imwrite("p1_inp.jpg",p1_inp)
cv2.imwrite("p2_inp.jpg",p2_inp)
print(sim(p1_inp, p2_inp,[0],0))
print(sim(p1_inp, p2_inp,[0],1))
print(sim(p1_inp, p2_inp,[0],2))
print(sim(p1_inp, p2_inp,[0],3))

print("sim_hash")
print(sim_hash(p1_, p2_))

print("sim0 3通道")
print(sim0(p1_inp, p2_inp,0))
print(sim0(p1_inp, p2_inp,1))
print(sim0(p1_inp, p2_inp,2))
print(sim0(p1_inp, p2_inp,3))

cv2.imwrite("p1_resize.jpg",cv2.resize(p1_inp,(8,8),interpolation=cv2.INTER_AREA))
cv2.imwrite("p2_resize.jpg",cv2.resize(p2_inp,(8,8),interpolation=cv2.INTER_AREA))

print("\ncvt_color:")
p1_cvt = cv2.cvtColor(cv2.resize(p1_inp, (8, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
p2_cvt = cv2.cvtColor(cv2.resize(p2_inp, (8, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
cv2.imwrite("p1_cvt.jpg",p1_cvt)
cv2.imwrite("p2_cvt.jpg",p2_cvt)
print(sim(p1_cvt, p2_cvt,[0],0))
print(sim(p1_cvt, p2_cvt,[0],1))
print(sim(p1_cvt, p2_cvt,[0],2))
print(sim(p1_cvt, p2_cvt,[0],3))

# def calcAndDrawHist(name,image):
#     hist = cv2.calcHist([image], [0], None, [16], [0, 256])
#     cv2.imshow(name,drawHist(hist))
# calcAndDrawHist("h1", cv2.imread("/Users/zac/5-Algrithm/python/CV/p1_cvt.jpg"))
# calcAndDrawHist("h2", cv2.imread("/Users/zac/5-Algrithm/python/CV/p2_cvt.jpg"))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

