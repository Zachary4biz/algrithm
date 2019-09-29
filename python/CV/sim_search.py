# author: zac
# create-time: 2019-09-29 12:01
# usage: -
import os
import cv2
from tqdm.auto import tqdm

# 注意缩小的时候AREA能避免类似摩尔纹的东西
def pre_format(pic):
    return cv2.resize(pic, (16, 16), interpolation=cv2.INTER_AREA)

def calHist(pic, bins=None):
    if bins is None:
        bins = [16]*3
    H1 = cv2.calcHist([pic], [0,1,2], None, bins, [0, 256, 0, 256, 0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)
    return H1

basePath = "/Users/zac/Downloads/img_sim_demo"

feature_list = []
featureImage_dir = os.path.join(basePath,"images")
for fName in tqdm(os.listdir(featureImage_dir), desc="calculating feature"):
    if fName.startswith("."):
        continue
    pic2 = cv2.imread(os.path.join(featureImage_dir, fName))
    feature_list.append([fName, calHist(pre_format(pic2))])


targetImage_dir = os.path.join(basePath,"pus_images")
for target_fName in os.listdir(targetImage_dir):
    if target_fName.startswith("."):
        continue
    print("at: {}".format(target_fName))
    pic = cv2.imread(os.path.join(targetImage_dir, target_fName))
    target_hist = calHist(pre_format(pic))
    res = []
    for search_fName, search_hist in tqdm(feature_list,desc="search for:{}".format(target_fName)):
        res.append([search_fName, round(cv2.compareHist(target_hist, search_hist, 2),4)])
    res.sort(key=lambda x: -x[1])
    print(target_fName + "\n" + "\n".join([str(i) for i in res[:3]]))
