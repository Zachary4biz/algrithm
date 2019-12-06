# author: zac
# create-time: 2019-10-28 16:50
# usage: -
from zac_pyutils import CVUtils,ExqUtils
from ImageFeature import Hist
import skimage.segmentation as seg
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm.auto import tqdm
import random


# YCbCr对"红"(第一通道)区分度很好，但是"绿蓝"(第二、三个通道)会比较集中于中间部分所以后两个增加几个bins
MODE="YCbCr"
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
        "http://img.boqiicdn.com/Data/BK/A/1908/9/imagick71411565338658_y.jpg"
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

    @staticmethod
    def random_samples(cnt=20):
        return random.sample(Samples.blue_birds+Samples.cartoon+Samples.white_cat+Samples.pyramid, cnt)


# 使用NN的最后一层作为图片的特征向量
def test_flow_vec_nn(img1, img_list, show=False, sortBy="total"):
    assert sortBy in ['total', 'color', 'lbp']
    sort_idx = ['total', 'color', 'lbp'].index(sortBy) + 2
    res = []
    transformer = CVUtils.Vectorize.VectorFromNN.InceptionV3()
    img1_vec = transformer.imgPIL2vec(img1)
    img_vec_list = transformer.imgPIL2vec_batch(img_list)
    for idx, img2_vec in tqdm(enumerate(img_vec_list), desc="sim_comp"):
        totalsim=CVUtils.cos_sim(img1_vec, img2_vec)
        res.append((idx, img_list[idx].resize(transformer.IMAGE_SHAPE), totalsim))
    res.sort(key=lambda x:x[sort_idx], reverse=True)
    if show:
        show_res = res
        row_num = int(math.pow(len(show_res), 0.5))
        col_num = row_num+1
        fig = plt.figure(figsize=(15, 15))
        fig.set_tight_layout(True)
        for plot_idx, (idx, img, totalsim) in tqdm(enumerate(show_res), desc="plot"):
            fig.add_subplot(row_num, col_num, plot_idx+1)
            plt.imshow(img)
            plt.axis("off")
            plt.text(x=0,y=img.size[1]+7,s=f"[idx]:{idx} [T]:{totalsim:.4f}",fontsize=6)
        fig.show()
    print(res)
    return res

# 整张图取主题色然后多个
def test_flow_nested_themecolor(img1, img_list, show=False, sortBy='total'):
    pass

# 多个子区域的直方图各自独立计算相似度然后取均值
def test_flow_nested_hist(img1, img_list, show=False, sortBy="total"):
    assert sortBy in ['total', 'color', 'lbp']
    sort_idx = ['total', 'color', 'lbp'].index(sortBy) + 2
    res = []
    for idx, img2 in enumerate(img_list):
        weight = np.array([[1,1,1,1,
                            1,2,2,1,
                            1,2,2,1,
                            1,1,1,1]]).flatten()
        totalsim=CVUtils.cos_sim_nested(Hist.PILImage.get_vector_nested(img1), Hist.PILImage.get_vector_nested(img2), weight=weight) # 0.8145783040892348
        histsim=CVUtils.cos_sim_nested(Hist.PILImage.get_vector_nested(img1, lbp_weight=0), Hist.PILImage.get_vector_nested(img2, lbp_weight=0), weight=weight) # 0.7991958782854643
        lbpsim=CVUtils.cos_sim_nested(Hist.PILImage.get_vector_nested(img1, color_weight=0), Hist.PILImage.get_vector_nested(img2, color_weight=0), weight=weight) # 0.9573060906603772
        res.append((idx, img2, totalsim, histsim, lbpsim))
    res.sort(key=lambda x:x[2], reverse=True)
    if show:
        show_res = res
        row_num = int(math.pow(len(show_res), 0.5))
        col_num = row_num+1
        fig = plt.figure(figsize=(15, 15))
        fig.set_tight_layout(True)
        for plot_idx, (idx, img, totalsim, histsim, lbpsim) in enumerate(show_res):
            if img.mode=='YCbCr':
                img = img.convert("RGB")
            fig.add_subplot(row_num, col_num, plot_idx+1)
            plt.imshow(img)
            plt.axis("off")
            plt.text(x=0,y=img.size[1]+7,s=f"[idx]:{idx} [T]:{totalsim:.4f} \n[CH]:{histsim:.2f} [LBPH]:{lbpsim:.2f}",fontsize=6)
        fig.show()
    print(res)
    return res

# 多个子区域的直方图拼接到一起计算相似度 | 效果不如nested取均值
def test_flow(img1, img_list, show=False, sortBy="total"):
    assert sortBy in ['total', 'color', 'lbp']
    sort_idx = ['total', 'color', 'lbp'].index(sortBy) + 2
    res = []
    for idx, img2 in enumerate(img_list):
        totalsim=Hist.General.cos_sim(Hist.PILImage.get_vector(img1), Hist.PILImage.get_vector(img2)) # 0.8145783040892348
        histsim=Hist.General.cos_sim(Hist.PILImage.get_vector(img1, lbp_weight=0), Hist.PILImage.get_vector(img2, lbp_weight=0)) # 0.7991958782854643
        lbpsim=Hist.General.cos_sim(Hist.PILImage.get_vector(img1, color_weight=0), Hist.PILImage.get_vector(img2, color_weight=0)) # 0.9573060906603772
        res.append((idx, img2, totalsim, histsim, lbpsim))
    res.sort(key=lambda x:x[2], reverse=True)
    if show:
        show_res = res
        row_num = int(math.pow(len(show_res), 0.5))
        col_num = row_num+1
        fig = plt.figure(figsize=(15, 15))
        fig.set_tight_layout(True)
        for plot_idx, (idx, img, totalsim, histsim, lbpsim) in enumerate(show_res):
            if img.mode=='YCbCr':
                img = img.convert("RGB")
            fig.add_subplot(row_num, col_num, plot_idx+1)
            plt.imshow(img)
            plt.axis("off")
            plt.text(x=0,y=img.size[1]+7,s=f"[idx]:{idx} [T]:{totalsim:.4f} \n[CH]:{histsim:.2f} [LBPH]:{lbpsim:.2f}",fontsize=6)
        fig.show()
    print(res)
    return res

# 展示一张图的hist、crops
def single_img_detail(img_inp):
    img_lbp = Hist.PILImage.get_img_lbp(img_inp)
    # weights = np.array([[1, 1, 1, 1],
    #                     [1, 2, 2, 1],
    #                     [1, 2, 2, 1],
    #                     [1, 1, 1, 1]])
    bins_color, bins_lbp = [32, 64, 32], [32]
    hist_rgb, crops_rgb = Hist.PILImage.get_hist(img_inp, row=4, col=4, bins=bins_color, return_crops=True)
    hist_lbp, crops_lbp = Hist.PILImage.get_hist(img_lbp, row=4, col=4, bins=bins_lbp, return_crops=True)
    Hist.General.plot_hist_crops(hist_rgb, crops_rgb).show()
    Hist.General.plot_hist_crops(hist_lbp, crops_lbp).show()

img1 = CVUtils.Load.image_by_pil_from(Samples.white_cat[0]).resize((64, 64)).convert(MODE)
img_list = [CVUtils.Load.image_by_pil_from(i).resize((64, 64)).convert(MODE).rotate(0) for i in Samples.random_samples(30)]
# img_list = [CVUtils.Load.image_by_pil_from(i).resize((64, 64)).convert(MODE).rotate(0) for i in Samples.blue_birds]
# test_flow(img1=img1,img_list=img_list,show=True)
# test_flow_nested_hist(img1=img1,img_list=img_list,show=True)
test_flow_vec_nn(img1=img1.convert("RGB"),img_list=[img.convert("RGB") for img in img_list],show=True)
# single_img_detail(img_list[2])

plt.show()
exit(0)

# CBIR
# 1.
import pickle
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

# 2.
with open("/Users/zac/Downloads/img_vec_inceptionV3.pck","rb") as frb:
    info = pickle.load(frb)

def get_fig(target):
    res = sorted([i+(CVUtils.cos_sim(target[2], i[2]),) for i in info if i[2] is not None], key=lambda x: x[-1], reverse=True)
    fig = plt.figure(figsize=(10,10))
    fig.set_tight_layout(True)
    fig.add_subplot(3,5,1)
    plt.axis("off")
    plt.imshow(np.array(CVUtils.Load.image_by_pil_from(target[1])))
    for idx,sim_target in enumerate(res[:14]):
        img = CVUtils.Load.image_by_pil_from(sim_target[1]).resize((299,299))
        fig.add_subplot(3,5, idx+2)
        plt.axis("off")
        plt.imshow(np.array(img))
        plt.text(x=0, y=img.size[1]+65, s=f"[id]: {sim_target[0]}\n[sim]:{sim_target[3]:.3f}", fontsize=9)
    return fig, res

for target in tqdm(info):
    fig,res = get_fig(target)
    fig.savefig("/Users/zac/Downloads/sim_samples/{}.png".format(target[0]))
    with open("/Users/zac/Downloads/sim_samples/sim_info.csv", "a") as fa:
        fa.writelines(target[0]+","+"|".join([i[0] for i in res])+"\n")

exit(0)




# for idx, url in enumerate(urls):


exit(0)

with open("/Users/zac/Downloads/picku_materials.csv","r") as f:
    urls = [i.strip().split(",")[1] for i in f.readlines()[1:]]





