# author: zac
# create-time: 2019-09-29 13:23
# usage: - 
import numpy as np
from PIL import Image
from functools import reduce
import os
from tqdm.auto import tqdm
import cv2
import pickle

# lbp
def tolbp(pic):
    pic_gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    row,col = pic_gray.shape
    for i in range(row):
        for j in range(col):
            center = pic_gray[i][j]
            pic_gray[i-1][j-1]




# histogram
def pre_format(pic):
    return cv2.resize(pic, (8, 8), interpolation=cv2.INTER_AREA)

def calHist(pic, bins=None):
    if bins is None:
        bins = [8]*3
    H1 = cv2.calcHist([pic], [0,1,2], None, bins, [0, 256, 0, 256, 0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)
    return H1

# hash
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

def hamming(h1, h2):
    h, d = 0, h1 ^ h2
    while d:
        h += 1
        d &= d - 1
    return h



basePath = "/Users/zac/Downloads/img_sim_demo"

#
# feature_list = []
# featureImage_dir = os.path.join(basePath,"images")
# for fName in tqdm(os.listdir(featureImage_dir), desc="calculating feature"):
#     if fName.startswith("."):
#         continue
#     pic = cv2.imread(os.path.join(featureImage_dir, fName))
#     feature_list.append([fName,  # file
#                          calHist(pre_format(pic)),  # hist
#                          phash(os.path.join(featureImage_dir, fName)),  # hash
#                          ])

# with open(os.path.join(basePath,"feature_list.pkl"),"wb+") as fwb:
#     pickle.dump(feature_list, fwb)
#     print("pkl obj saved.")

with open(os.path.join(basePath,"feature_list.pkl"),"rb+") as frb:
    feature_list_ = pickle.load(frb)

from zac_pyutils import ExqLog
logger = ExqLog.get_file_logger(info_log_file=os.path.join(basePath, "result.out"),
                                err_log_file=os.path.join(basePath, "result_err.out"))

similarity_result = {}
targetImage_dir = os.path.join(basePath,"pus_images")
for target_fName in tqdm(os.listdir(targetImage_dir)):
    if target_fName.startswith("."):
        continue
    # print("at: {}".format(target_fName))
    target_hash = phash(os.path.join(targetImage_dir, target_fName))
    pic = cv2.imread(os.path.join(targetImage_dir, target_fName))
    target_hist = calHist(pre_format(pic))
    res = []
    for search_fName, search_hist, search_hash in feature_list_:
        hist_distance = round(cv2.compareHist(search_hist,target_hist,method=1),4)
        hash_distance = hamming(search_hash, target_hash)
        final_distance = round((hash_distance**2 + hist_distance**2)**0.5,4)
        res.append([search_fName,final_distance,hist_distance,hash_distance])
    res.sort(key=lambda x: x[1])
    logger.info("\n"+ target_fName + "\n" + "\n".join([str(i) for i in res[:3]])+"\n")
    similarity_result.update({target_fName:res[:3]})
    # if res[0][1] <= 5: # 最接近的hash小于5就输出top3
    #     print(target_fName + "\n" + "\n".join([str(i) for i in res[:3] if i[1]<=5]))

with open(os.path.join(basePath,"similarity_result.pkl"),"wb+") as fwb:
    pickle.dump(similarity_result,fwb)


import subprocess
def openit(target,sim_list):
    openTargetImg = "open /Users/zac/Downloads/img_sim_demo/pus_images/{}".format(target)
    openSearchImgs = ["open /Users/zac/Downloads/img_sim_demo/images/{}".format(i) for i in sim_list]
    subprocess.getstatusoutput(openTargetImg)
    for i in openSearchImgs:
        subprocess.getstatusoutput(i)

with open("/Users/zac/Downloads/img_sim_demo/similarity_result.pkl","rb+") as frb:
    info = pickle.load(frb)

for target,simInfo_list in info.items():
    if 2 < simInfo_list[0][1] <= 3:
        print(">> found one:")
        print(target)
        for i in simInfo_list: print(i)
        openit(target, [i[0] for i in simInfo_list])
        inp = input("press 'g' to continue")
        if inp == 'g':
            continue
