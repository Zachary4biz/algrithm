# author: zac
# create-time: 2019-11-12 18:33
# usage: - 


# KMeans 提取主题色
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import numpy as np
from zac_pyutils import CVUtils
import seaborn as sns
import time

imgPIL = CVUtils.Load.image_by_pil_from("http://pic1.win4000.com/wallpaper/0/57ba5b70305e0.jpg")
imgPIL2 = CVUtils.Load.image_by_pil_from("https://androidhuilin.wang/img/pictures/picture2.jpg")
imgPIL3 = CVUtils.Load.image_by_pil_from("http://jiliuwang.net/wp-content/uploads/2018/12/20181229021035_68235.jpg")
class KMeansColor:
    def __init__(self, cluster=5):
        self.km = MiniBatchKMeans(n_clusters=cluster, random_state=0)

    def imgPIL2Vec(self, imgPIL):
        img_arr = np.array(imgPIL)
        return self.imgArr2Vec(img_arr)

    def imgPIL2VecWeight(self, imgPIL):
        img_arr = np.array(imgPIL)
        return self.imgArr2VecWeight(img_arr)

    def imgArr2Vec(self, imgArr):
        return self.imgArr2VecWeight(imgArr)[0]

    def imgArr2VecWeight(self, imgArr):
        """
        返回结果是 (色彩聚类的中心, 该类的占比)
        """
        pixel = np.reshape(imgArr, (-1, imgArr.shape[2]))
        km_res = self.km.fit_predict(pixel)
        center_ratio = np.array([np.append(self.km.cluster_centers_[centerIdx],len(km_res[km_res==centerIdx]) / len(pixel)) for centerIdx in np.sort(np.unique(km_res))])
        center_ratio = center_ratio[center_ratio[:, 3].argsort()][::-1]  # 按第三列降序
        return center_ratio[:,:3], center_ratio[:,3]

tsf = KMeansColor(cluster=5)
# tsf = CVUtils.Vectorize.VectorFromThemeColor.KMeansColor(cluster=5)
begin = time.time()
theme,w = tsf.imgPIL2VecWeight(imgPIL)
theme2,w2 = tsf.imgPIL2VecWeight(imgPIL2)
theme3,w3 = tsf.imgPIL2VecWeight(imgPIL3)
sim = CVUtils.cos_sim_nested(theme, theme2)
print(f"耗时：{time.time() - begin}, 相似度: {sim}")
imgPIL.show()
imgPIL2.show()
imgPIL3.show()
sns.palplot(theme/255.0)
sns.palplot(theme2/255.0)
sns.palplot(theme3/255.0)
plt.show()



