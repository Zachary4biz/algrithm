# encoding=utf-8
from PCV.tools import imtools
import pickle
import scipy
from pylab import *
from PIL import Image
from scipy.cluster.vq import *
from PCV.tools import pca




img_w = 128
img_h = 128
img_deep = 3
pca_demension = 40 # 亦即取40张图去和特征做dot点乘
k_category = 8 # k-means聚类,设置为8类

#
imgs_path = "/Users/zac/5-Algrithm/algrithm-data/ImageCluster/images_resized"
imlist = imtools.get_imlist(imgs_path)

# 获取图像列表和其尺寸
im = array(Image.open(imlist[0])) # open one image to get the size
m, n = im.shape[:2] # get the size of the images
imnbr = len(imlist) # get number of images
print(" the number of images is %d" % imnbr)

# Create matrix to store all flattened images
immatrix = array([array(Image.open(imname)).flatten() for imname in imlist], 'f')

# PCA降维
V, S, immean = pca.pca(immatrix)
# 保存均值和主成分
pkl_path = "/Users/zac/5-Algrithm/algrithm-data/ImageCluster/pca_model.pkl"
#f = open('./a_pca_modes.pkl', 'wb')
f = open(pkl_path, 'wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()

with open(pkl_path,'rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)
# create matrix to store all flattened images
immatrix = array([array(Image.open(im)).flatten() for im in imlist],'f')


# project on the 40 first PCs
immean = immean.flatten()
projected = array([dot(V[:30],immatrix[i]-immean) for i in range(imnbr)])

# k-means
projected = whiten(projected)
centroids,distortion = kmeans(projected,8)
code,distance = vq(projected,centroids)

# plot clusters
for k in range(8):
    ind = where(code==k)[0]
    figure()
    gray()
    for i in range(minimum(len(ind),40)):
        subplot(4,10,i+1)
        # imshow(immatrix[ind[i]].reshape((600,600,3)))
        new_im=Image.fromarray(immatrix[ind[i]].reshape((600,600,3)).astype(np.uint8))
        new_im.save("/Users/zac/5-Algrithm/algrithm-data/ImageCluster/images_clustered/%s-%s.jpg"%(k,i))
        axis('off')
show()
