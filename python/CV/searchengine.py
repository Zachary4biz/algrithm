import sys
sys.path.append("/Users/zac/5-Algrithm/python/CV")

from colordescriptor import ColorDescriptor
from searcher import Searcher,chi2_distance
import argparse
import cv2
import os
from tqdm.auto import tqdm
import pickle

cd = ColorDescriptor((8, 12, 3))

feature_list = []
featureImage_dir = "/Users/zac/Downloads/img_sim_demo/images/"
for f in tqdm(os.listdir(featureImage_dir)):
    if f.startswith("."):
        continue
    pic2 = cv2.imread(os.path.join(featureImage_dir, f))
    f2 = cd.describe(pic2)
    feature_list.append([f,f2])

with open("/Users/zac/Downloads/img_sim_demo/images_feature.pkl","wb+") as fwb:
    pickle.dump(feature_list,fwb)

with open("/Users/zac/Downloads/img_sim_demo/images_feature.pkl","rb+") as frb:
    feature_list_ = pickle.load(frb)


targetImage_dir = "/Users/zac/Downloads/img_sim_demo/pus_images/"
for f in os.listdir(targetImage_dir):
    if f.startswith("."):
        continue
    pic = cv2.imread(os.path.join(targetImage_dir, f))
    f1 = cd.describe(pic)
    res = []
    print("    searching ...")
    for k, v in tqdm(feature_list_):
        res.append((k, chi2_distance(f1, v)))
    res.sort(key=lambda x: x[1])
    print(f+"\n"+"\n".join([str(i) for i in res[:3]]))

# pic1 = cv2.imread("1569638480188506.jpg")
# f1 = cd.describe(pic1)
# res = []
# for k,v in tqdm(feature_list):
#     res.append((k,chi2_distance(f1,v)))
#
# res.sort(key=lambda x: x[1])
# print(res[0])



assert False
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True, 
    help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
    help = "Path to the query image")
ap.add_argument("-d", "--dataset", required = True,
    help = "Path to the result path")
args = vars(ap.parse_args())
#上面这一段类似于index中，用于读取命令行的指令，解决三个路径：
#图像库中整理好提取出的所有图像特征之记录在index.csv的路径、待查询图像的路径、图像库在哪里
 
 
cd = ColorDescriptor((8, 12, 3))

query = cv2.imread(args["query"])
features = cd.describe(query)
print(args["query"])
print(query)
searcher = Searcher(args["index"])
results = searcher.search(features)
print ("this is %s" % results)
#上述为显示一下计算相似度的结果，值越小的即越相似
cv2.imshow("Query", query)
 
for (score, resultID) in results:
    result = cv2.imread(args["dataset"] + "/" + resultID)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyWindow("Result")
 
cv2.waitKey(0)
cv2.destroyAllWindows()
 
 
''' result = cv2.imread(r"H:\result\1.png")
    print("score is %s" % score)
    print("resultID is %s" % resultID)
    print("this part is___________%s" % args["dataset"])
    print("result is %s" % result)
'''
