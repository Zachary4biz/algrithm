# author: zac
# create-time: 2019-11-08 10:25
# usage: - 
"""
直接抽inceptionV3的最后一层做图像特征,2048维
向量聚类
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from zac_pyutils import CVUtils
from tqdm.auto import tqdm
import random
import time
import matplotlib.patches as patches
import itertools
from sklearn import metrics
from sklearn.cluster import *
import math

class Samples:
    blue_birds = [
        "http://www.kedo.gov.cn/upload/resources/image/2017/04/24/150703.png",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3Wm23HuKYuKMiSo9U_UAFDYc1_ccodPS9PMNrOWesI3lAE0bF&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQythkV6vmH4FnVuiJFkPAnj-_iAca42bMf1eZQDEGKEnO5zMzC&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThwIfzyp-Rv5zYM0fwPmoM5k1f9eW3ETYuPcL8j2I0TuG0tdb5&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzzG3lfGmXZJN2OPQvxRTTLqwMvVIaHd-BVrC88FQvtpUuMidR&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSOT85WhH3PJ7VNA64bC3bmm_wH3UUt33xHOaT8Mc7unj1F_p0l&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSG5gPEotCi2WMBj5lyh8ftdRwjiySyViFeu-eB24bIKW_SFOrP&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTq6-juHAdZNKHHDdBNZ5fWGywrVxhLGFEdLE6mcSo5WdsPNw9_&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAoKVoo9HZFqWWfuJDKGGmPybhj99pn2kVcplJFZbtyrE9csFI&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-i6UBlc4TbDH85S4LYh9rYvdE-F9eZX0Azr0n4ySSRxBtPPnd&s",
        # "http://img.boqiicdn.com/Data/BK/A/1908/9/imagick71411565338658_y.jpg" # cat
    ]
    cartoon = [
        "https://png.pngtree.com/png-clipart/20190617/original/pngtree-ancient-characters-q-version-moe-illustration-png-image_3834249.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWzxDGvkniSg2uYGB1lY06cicXM2tsjrYDRE1GwoS9rBJQHPP7&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYs1FUZTewRyaceTlP8nAEX0jI9zLM2Z_0o9zdDT26FkINsI9r&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScY_5WL6lFFP510ZZBlKbbUTi1oxw79XZOU3KwrB5-clkZ4o7J&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT_g4up-VwBIP0UESG3qEAXvhng_-rQhKbBPs1zlFvh28bjCXdm&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSLzZPSSo0OXvSptcIZkiq_8HkHiFr7gOcYZHfnjeq0jtBD4_dh&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4OMEvJnADiAV0wST0dTdaRJHIFSwrG9L-U47lakXpq1wiwekx&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSmhOowW42bpCwh9hofbJjsINh3MSQm_c4ZgNI5mnXKT33u81uL&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGI2N9hko1pymx_EoarfVXxBwz7fBETPwcLDD_qG1Htq65zJ_K&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQbwmMandg0e1VZl_BGW8hJlMD5bItcgcRrUX1cQtwEl-mMqvTp&s"
    ]
    white_cat = [
        "https://pic2.zhimg.com/v2-0f46e56eb41906c6969540478bae5184_1200x500.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLd7WnZCNZV0da1Urw9wDg2HtlfK6PUBYKSw4lKllmGPHe-AHU&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjKcBZjcyToc0rhlWTEIF8p49jh05munfP3q61t5o_Hj6Zygit&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRf6ZjHx5CvBOSAGSWdKnT-pNl2uItbUSV181LLFAb46Y6c7CFq&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFDarJmcFqwn6wNFOKE971bzZMz6jRFbuOUbsic_6sx7F-O6M0&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTqkTyLYyP_4B1SftKYaEn-gGlxHpBUySNxqKkIlCiX09n2MLX7&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSt78JI12oeg6haidCNAUhSAhbJXfKfQpw6o98G45H-SSqPVlIK&s"
    ]
    pyramid = [
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSXb-J88mYMC6Zy4hsGv315xANFBy-cyXG0iIwMERSVoB1iP9Ej&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThcdt1nV1YElMCPhQVVKLo3cb7ZdE_gZmjMfVP0vX7dnDqNMSd&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREBvRy6kXUKtdZFfDdD_s1UZd8LorkudYfgO0QzMF89UDXx_ih&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQO_6IOgZtWwCxtj0q1BnDdZwYd_ScDUcbXQ5VPeIhk6ZoSNbqm&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQkCpsOKFGhJATu6ucg3ayeGtS9rWkpTejHS0GJfn8cncG_k-Z&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTfauaep8uZMRxzaGXp7M_vhMB4s52wGKdk8DHW_dcFpG7j4Jp6&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRDBamzIg-jEbblkDNGXkWTE7lblIlpOnodJ9oMZYkU86rE1PKv&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRLnAwZ-PIQ-T8Dgl1Wvk6YKTQW77aqLRkMSJZnSEK4IAB6dhdF&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRkzthzjRc5iacNhmof9Jfywav7cnpsVdWXhw_8jzW7EZeow_10&s",
    ]
    all = blue_birds + cartoon + white_cat + pyramid
    all_nested = [blue_birds, cartoon, white_cat, pyramid]

    @staticmethod
    def random_samples(cnt=20):
        return random.sample(Samples.blue_birds + Samples.cartoon + Samples.white_cat + Samples.pyramid, cnt)


# https://www.rapidtables.com/web/color/RGB_Color.html
rgb_color = {
    "maroon": (128, 0, 0), "dark red": (139, 0, 0), "brown": (165, 42, 42), "firebrick": (178, 34, 34), "crimson": (220, 20, 60),
    "tomato": (255, 99, 71), "coral": (255, 127, 80), "indian red": (205, 92, 92), "light coral": (240, 128, 128), "dark salmon": (233, 150, 122),
    "salmon": (250, 128, 114), "light salmon": (255, 160, 122), "orange red": (255, 69, 0),
    "dark orange": (255, 140, 0), "orange": (255, 165, 0), "gold": (255, 215, 0), "dark golden rod": (184, 134, 11), "golden rod": (218, 165, 32),
    "pale golden rod": (238, 232, 170), "dark khaki": (189, 183, 107), "khaki": (240, 230, 140), "olive": (128, 128, 0),
    "yellow green": (154, 205, 50),
    "dark olive green": (85, 107, 47), "olive drab": (107, 142, 35), "green yellow": (173, 255, 47),
    "dark green": (0, 100, 0), "forest green": (34, 139, 34), "lime green": (50, 205, 50),
    "light green": (144, 238, 144), "pale green": (152, 251, 152), "dark sea green": (143, 188, 143), "medium spring green": (0, 250, 154),
    "spring green": (0, 255, 127), "sea green": (46, 139, 87), "medium aqua marine": (102, 205, 170), "medium sea green": (60, 179, 113),
    "light sea green": (32, 178, 170),
    "dark slate gray": (47, 79, 79), "teal": (0, 128, 128), "dark cyan": (0, 139, 139),
    "dark turquoise": (0, 206, 209), "turquoise": (64, 224, 208), "medium turquoise": (72, 209, 204),
    "pale turquoise": (175, 238, 238), "aqua marine": (127, 255, 212), "powder blue": (176, 224, 230), "cadet blue": (95, 158, 160),
    "steel blue": (70, 130, 180), "corn flower blue": (100, 149, 237), "deep sky blue": (0, 191, 255), "dodger blue": (30, 144, 255),
    "light blue": (173, 216, 230), "sky blue": (135, 206, 235), "light sky blue": (135, 206, 250), "navy": (0, 0, 128),
    "royal blue": (65, 105, 225),
    "blue violet": (138, 43, 226), "indigo": (75, 0, 130), "dark slate blue": (72, 61, 139), "slate blue": (106, 90, 205), "medium slate blue": (123, 104, 238),
    "medium purple": (147, 112, 219), "dark magenta": (139, 0, 139), "dark violet": (148, 0, 211), "dark orchid": (153, 50, 204),
    "medium orchid": (186, 85, 211), "purple": (128, 0, 128), "thistle": (216, 191, 216), "plum": (221, 160, 221), "violet": (238, 130, 238),
    "magenta / fuchsia": (255, 0, 255), "orchid": (218, 112, 214), "medium violet red": (199, 21, 133), "pale violet red": (219, 112, 147),
    "deep pink": (255, 20, 147), "hot pink": (255, 105, 180), "light pink": (255, 182, 193), "pink": (255, 192, 203),
    "antique white": (250, 235, 215), "beige": (245, 245, 220), "bisque": (255, 228, 196), "blanched almond": (255, 235, 205), "wheat": (245, 222, 179),
    "corn silk": (255, 248, 220), "lemon chiffon": (255, 250, 205), "light golden rod yellow": (250, 250, 210), "light yellow": (255, 255, 224),
    "saddle brown": (139, 69, 19), "sienna": (160, 82, 45), "chocolate": (210, 105, 30), "peru": (205, 133, 63), "sandy brown": (244, 164, 96),
    "burly wood": (222, 184, 135), "tan": (210, 180, 140), "rosy brown": (188, 143, 143), "moccasin": (255, 228, 181), "navajo white": (255, 222, 173),
    "peach puff": (255, 218, 185),
    "misty rose": (255, 228, 225), "papaya whip": (255, 239, 213), "slate gray": (112, 128, 144), "light steel blue": (176, 196, 222)
}

def get_color(totalCnt=10):
    maxStep = len(rgb_color.values()) // totalCnt
    random.randint(3, maxStep)
    res = [[j / 255.0 for j in list(rgb_color.values())[i]]
     for i in range(0, len(rgb_color.values()), maxStep)]
    # sns.palplot(res)
    return res

###########
# 降维处理
###########
def init_inceptionV3_vectors():
    a = input("re-init inceptionV3? y/n")
    if a == "y":
        with open("/Users/zac/Downloads/picku_materials.csv", "r") as fr:
            ids_urls=[i.strip().split(",")[:2] for i in fr.readlines()][1:]
        transformer = CVUtils.Vectorize.VectorFromNN.InceptionV3()
        res = []
        for (id_, url) in tqdm(ids_urls):
            try:
                img = CVUtils.Load.image_by_pil_from(url).convert("RGB")
                res.append((id_, url, transformer.imgPIL2vec(img)))
            except Exception:
                res.append((id_, url, None))
        with open("/Users/zac/Downloads/img_vec_inceptionV3.pck","wb+") as fwb:
            pickle.dump(res,fwb)
# init_inceptionV3_vectors()
with open("/Users/zac/Downloads/img_vec_inceptionV3.pck", "rb") as frb:
    img_vec = pickle.load(frb)
    img_vec_dict = {int(i[0]):i[1:] for i in img_vec}
# 每次画一个聚类的簇
def show_tsne_animation(emb, c=None, name=None):
    # 色彩
    if c is None:
        c = [random.choice([0]) for _ in range(len(emb))]
    groupeIter=itertools.groupby(sorted([(idx,rgb) for idx, rgb in enumerate(c)], key=lambda x:x[1]), key=lambda x:x[1])
    color_embIdx = [(k,[idx for idx,rgb in g]) for k,g in groupeIter]

    # 归一化
    x_min, x_max = np.min(emb, 0), np.max(emb, 0)
    emb = emb / (x_max - x_min)

    # plot
    fig = plt.figure(name)
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d(left=-1,right=1)
    ax.set_ylim3d(bottom=-1,top=1)
    ax.set_zlim3d(bottom=-1,top=1)
    # ax.set_axis_off()

    # 动画
    def animate(i):
        print(i)
        color, embIdx = color_embIdx[i]
        emb_part = np.array([emb[idx] for idx in embIdx])
        ax.text3D(x=1, y=-1, z=0, s=f"at {i}")
        ax.scatter(xs=emb_part[:, 0], ys=emb_part[:, 1], zs=emb_part[:, 2], c=color)
        return ax
    def init():
        # ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c='gray')  # 灰色的点一直都在没有被覆盖掉，旋转的时候才能发现
        return ax

    ani = animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=len(color_embIdx), interval=3*1000, blit=False, repeat=False)

    return ani

# 每次画一个/十个点
def show_tsne_animation_each_point(emb, c=None, name=None, batch=50, rotate_speed=1, axis_off=False):
    # 色彩
    if c is None:
        c = [random.choice([0]) for _ in range(len(emb))]
    embIdx_color=sorted([(idx,rgb) for idx, rgb in enumerate(c)], key=lambda x:x[1])

    # 归一化
    x_min, x_max = np.min(np.abs(emb), 0), np.max(np.abs(emb), 0)
    emb = emb / (x_max - x_min + 1)  # +1解决一下浮点数除法得到1.000043的情况（控制不超过1）

    # plot
    fig = plt.figure(num=name)
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d(left=-1,right=1)
    ax.set_ylim3d(bottom=-1,top=1)
    ax.set_zlim3d(bottom=-1,top=1)
    if axis_off:
        ax.set_axis_off()

    def init():
        # ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c='gray')  # 灰色的点一直都在没有被覆盖掉，旋转的时候才能发现
        return ax

    def animate(i):
        # 输入的i是画（按类排序后的）**第几**个点
        # 每次画的这batch个点都要用 j 去取排好序的点的信息embIdx_color
        p_to_draw = embIdx_color[i*batch : (i+1)*batch]
        p_to_draw_idx = np.array([emb[idx] for idx,color in p_to_draw])
        p_to_draw_color = [color for idx,color in p_to_draw]
        ax.scatter(xs=p_to_draw_idx[:, 0], ys=p_to_draw_idx[:, 1], zs=p_to_draw_idx[:, 2], c=p_to_draw_color)
        ax.view_init(elev=ax.elev+rotate_speed, azim=ax.azim+rotate_speed)

        return ax

    ani = animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=len(embIdx_color) // batch + 1, interval=0.01*1000, blit=False, repeat=False)

    return ani

# 展示每个类别下的多少张图
def show_pictures_clustered():
    plt.imshow()

def show_tsne(emb, c=None, name=None):
    if c is None:
        c = [random.choice([0]) for _ in range(len(emb))]
    set(str(i) for i in c)
    # 归一化
    x_min, x_max = np.min(emb, 0), np.max(emb, 0)
    emb = emb / (x_max - x_min)

    # plot
    fig = plt.figure(name)
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_axis_off()

    ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=c)
    fig.show()
    return fig

# def show_cmap(cmap):
#     fig = plt.figure(figsize=(9,2))
#     ax = fig.add_subplot(111)
#     ax.set_title(f"in total {total_color_cnt} colors")
#     ax.set_xlim((0, total_color_cnt))
#     ax.set_ylim((0, 0.5))
#     for idx, color in enumerate(c_set):
#         w,h=0.2,1
#         rect = plt.Rectangle(xy=(idx,0),width=w, height=h, facecolor=color)
#         print(rect)
#         ax.add_artist(rect)
#         plt.text(idx+w,0, s="\n".join([f"{i:.3f}" for i in color]), fontsize=6)
#     fig.show()


# 降维至3 | (3400, 2048) --> (3400, 3) 耗时约五分钟
# emb = TSNE(n_components=3).fit_transform([i[-1] for i in img_vec if i[-1] is not None])
# with open("/Users/zac/Downloads/img_vec_inceptionV3_emb.pck","wb+") as fwb:
#     pickle.dump(emb, fwb)
with open("/Users/zac/Downloads/img_vec_inceptionV3_emb.pck", "rb+") as frb:
    tsne_emb = pickle.load(frb)

show_tsne(tsne_emb, name="tsne_emb")

# 降维至3 | Samples类里的37条数据， (37, 2048) --> (37, 2049) 两秒
# transformer = CVUtils.Vectorize.VectorFromNN.InceptionV3()
# sample_idx_img_list = [(idx, CVUtils.Load.image_by_pil_from(url).convert("RGB")) for idx, category in tqdm(enumerate(Samples.all_nested)) for url in category]
# sample_vec_list = list(zip([i[0] for i in sample_idx_img_list], transformer.imgPIL2vec_batch([i[1] for i in sample_idx_img_list])))
# sample_emb = TSNE(n_components=3).fit_transform([vec for idx,vec in sample_vec_list])
# cmap = plt.get_cmap('brg', len(set([idx for idx,vec in sample_vec_list])))
# show_tsne(sample_emb, c=[cmap(idx) for idx, vec in sample_vec_list], name="")

#######
# 聚类
#######

def do_cluster(obj,X):
    begin = time.time()
    clusters = obj.fit_predict(X)
    sc_score = metrics.silhouette_score(X=X, labels=clusters, metric="cosine")
    ch_score = metrics.calinski_harabaz_score(X=X, labels=clusters)
    cm = get_color(np.unique(clusters).size)
    ani = show_tsne_animation_each_point(tsne_emb, c=[cm[cla] for cla in clusters],
                                         name=f"{type(obj).__name__}_{np.unique(clusters).size}", batch=30,
                                         rotate_speed=2, axis_off=True)
    print(f"{type(obj).__name__} sc: {sc_score:.4f} ch: {ch_score:.4f} elapse: {time.time() - begin}")
    return clusters, ani

# 生成三种聚类方式的gif图，好玩用的
def total_animation():
    img_vec_ready = np.array([i[-1] for i in img_vec if i[-1] is not None])
    kmeans = KMeans(n_clusters=10, random_state=0)
    dbscan = DBSCAN(eps=0.2, min_samples=10, metric="cosine")
    birch = Birch(n_clusters=10)
    res={}
    for i in [kmeans, dbscan, birch]:
        cluster_res,ani = do_cluster(i, img_vec_ready)
        res.update({type(i).__name__:{"cluster_res":cluster_res, "ani":ani}})

    for k,v in res.items():
        v['ani'].save(f"/Users/zac/Downloads/ani_{k}.gif", fps=30, writer='imagemagick')

    plt.show()



# 每个聚类里随机抽几个出来看看
def serious_job():
    img_vec_ready = np.array([i[-1] for i in img_vec if i[-1] is not None])
    kmeans = MiniBatchKMeans(n_clusters=10, random_state=0)
    cluster_res = kmeans.fit_transform(img_vec_ready)
    print("聚类后的统计结果: ",np.unique(np.argmax(cluster_res, axis=1),return_counts=True))
    prob_range = np.max(cluster_res, axis=1) - np.min(cluster_res, axis=1) # 每行里概率最大与最小的两类取差
    print(f"聚类后，各样本（属于各类）的概率极差的：[avg]: {np.mean(prob_range):.3f} [median]: {np.median(prob_range):.3f} [max]: {np.max(prob_range):.3f} [min]: {np.min(prob_range):.3f}")
    check_class = 7
    near_centers_idx = np.intersect1d(np.where(np.argmax(cluster_res, axis=1)==check_class), np.where(prob_range > np.percentile(prob_range, 75)))
    img_vec_arr = np.array(img_vec)
    imgPIL_list = [show_by_id(int(i),plot=False) for i in img_vec_arr[near_centers_idx][:,0]]
    r = int(len(imgPIL_list) ** 0.5)
    c = len(imgPIL_list) // r + 1
    fig, axes_list = plt.subplots(r, c)
    for i in axes_list:
        for j in i:
            j.set_axis_off()
    for idx, img in enumerate(imgPIL_list):
        axes = axes_list[idx // c, idx % c]
        axes.set_axis_off()
        dis_to_center = prob_range[near_centers_idx][idx]
        axes.text(x=0, y=axes.get_ylim()[1] + 10, s=dis_to_center, fontsize=6)
        axes.imshow(np.array(img.resize((640, 640))))
    # idx_arr = np.where(cluster_res==8)[0]
    # # 可视化
    # r,c = 4,4
    # total=r*c
    # fig, axes_list = plt.subplots(r,c)
    # for idx, i in enumerate(np.array(img_vec)[np.random.choice(idx_arr, total)]):
    #     axes = axes_list[idx // c, idx % c]
    #     axes.set_axis_off()
    #     axes.text(x=0,y=axes.get_ylim()[1]+10,s=i[0], fontsize=6)
    #     axes.imshow(np.array(CVUtils.Load.image_by_pil_from(i[1]).resize((640, 640))))

def most_sim_by_id(id_, topN=10, plot=True):
    vec = img_vec_dict[id_][1]
    res = sorted([(k,CVUtils.cos_sim(vec,v[1])) for k,v in img_vec_dict.items() if v[1] is not None], key=lambda x: -x[1])[:topN]
    if plot:
        r = int(topN ** 0.5)
        c = (topN // r) + 1
        fig, axes_list = plt.subplots(r, c)
        for idx, (id_, sim) in enumerate(res):
            axes = axes_list[idx // c, idx % c]
            axes.set_axis_off()
            axes.text(x=0, y=axes.get_ylim()[1] + 10, s=f"[{id_}]: {sim:.3f}", fontsize=6)
            axes.imshow(np.array(CVUtils.Load.image_by_pil_from(img_vec_dict[id_][0]).resize((640, 640))))
    return res

def show_by_id(id_, plot=True):
    url = img_vec_dict[id_][0]
    if plot:
        CVUtils.Load.image_by_pil_from(url).show()
    return CVUtils.Load.image_by_pil_from(url)



total_animation()
