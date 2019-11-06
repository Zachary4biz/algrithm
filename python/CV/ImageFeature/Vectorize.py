# author: zac
# create-time: 2019-11-06 14:18
# usage: - 

import sys
import importlib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image
import logging
logging.disable(logging.WARNING)

def _get_module(name):
    # return sys.modules.get(name, default=__import__(name))
    return sys.modules.get(name, importlib.import_module(name))

class VectorFromNN:
    class inceptionV3:
        url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
        IMAGE_SHAPE = (299, 299)  # inception_v3 强制shape是299x299
        @staticmethod
        def pre_format_pilImage(imgPIL):
            return np.array(imgPIL.resize(VectorFromNN.inceptionV3.IMAGE_SHAPE)) / 255.0

    default_model = None
    def get_default_model(self):
        if self.default_model is None:
            self.default_model = VectorFromNN.init_inceptionV3()
        return self.default_model

    @staticmethod
    def init_inceptionV3():
        f_vec = tf.keras.Sequential([hub.KerasLayer(VectorFromNN.inceptionV3.url, output_shape=[2048], trainable=False)])
        return f_vec

    def imgPIL2vec(self, imgPIL, model=None):
        imgArr = VectorFromNN.inceptionV3.pre_format_pilImage(imgPIL)
        return self.imgArr2vec(imgArr,model=model)

    def imgPIL2vec_batch(self, imgPIL_batch, model=None):
        imgArr_batch = np.array([VectorFromNN.inceptionV3.pre_format_pilImage(imgPIL) for imgPIL in imgPIL_batch])
        return self.imgArr2vec_batch(imgArr_batch,model=model)

    def imgArr2vec(self, imgArr, model=None):
        return self.imgArr2vec_batch(imgArr[np.newaxis, :], model=model)[0]

    def imgArr2vec_batch(self, imgArr_batch, model=None):
        if model is None:
            model = self.get_default_model()
        return model.predict(imgArr_batch)

class StandardCV:
    @staticmethod
    def custom_cut_to_matrix(imgPIL, row, col):
        # 0 1 2
        # 3 4 5
        # 6 7 8
        #    col0       col1       col2       col3
        # (  0,  0)  (1/3,  0)  (2/3,  0)  (3/3,  0)   row0
        # (  0,1/3)  (1/3,1/3)  (2/3,1/3)  (3/3,1/3)   row1
        # (  0,2/3)  (1/3,2/3)  (2/3,2/3)  (3/3,2/3)   row2
        # (  0,3/3)  (1/3,3/3)  (2/3,3/3)  (3/3,3/3)   row3
        crop_matrix = []
        base_w, base_h = imgPIL.size[0] // col, imgPIL.size[1] // row
        for r in range(row):
            crop_one_row = []
            for c in range(col):
                anchor_lt = (base_w * c, base_h * r)  # 左上角的anchor
                anchor_rb = (base_w * (c + 1), base_h * (r + 1))  # 右下角的anchor
                crop_one_row.append(imgPIL.crop((anchor_lt[0], anchor_lt[1], anchor_rb[0], anchor_rb[1])))
            crop_matrix.append(crop_one_row)
        return np.array(crop_matrix, dtype='object')

    @staticmethod
    def get_hist(imgPIL, row=1, col=1, bins=None, weights=None, return_crops=False):
        """
        输入要求是PIL的image
        根据 row,col 会切分图像，最后返回时 all_crops 的size是(row, col, img_w, img_h, img_channel)
        各区域单独计算hist然后拼接到一起
        :param bins: 设定各个通道的bins个数，默认[32]*3。如果不是一样大的会padding到和max一样大
                     例如R通道4个bins: [16,16,16,32]，G通道2个bins：[32,48]，G会padding为[32,48,0,0]
        :param weights: 各个区域hist权重，默认都为1
        """
        if bins is None:
            bins = [32] * 3
        if weights is None:
            weights = [1.0] * row * col
        is_gray = len(np.array(imgPIL).shape) == 2
        # 切块
        all_crops = StandardCV.custom_cut_to_matrix(imgPIL, row, col)  # come with row x col
        all_crops_flatten = np.concatenate(all_crops, axis=0)  # flattent to (16, 128, 128, 3)

        hist_features = []
        for cropIdx, cropPIL in enumerate(all_crops_flatten):
            cropArr = np.array(cropPIL)
            # np.array(crop).shape[-1] 取通道数
            # np.histogram 返回结果两个， hist, bins_edge —— 直方图和bin的方位
            if is_gray:
                # 灰度图的hist为了方便后面统一使用，也在外面包上一个维度
                hist_of_all_channel = [np.histogram(cropArr, bins=bins[0], range=[0, 256])[0]]
            else:
                hist_of_all_channel = [np.histogram(cropArr[:, :, c], bins=bins[c], range=[0, 256])[0] for c in range(cropArr.shape[-1])]
            # normalize
            hist_of_all_channel = [hist / sum(hist) for hist in hist_of_all_channel]
            # 不同通道bins可能有不同，末尾padding到相同
            # 例如R通道4个bins: [16,16,16,32]，G通道2个bins：[32,48]，G会padding为[32,48,0,0]
            max_bin = max(bins)
            hist_of_all_channel = [np.pad(hist, (0, max_bin - bins[idx]), 'constant') for idx, hist in enumerate(hist_of_all_channel)]
            # 此区域三通道的hist都乘上权重
            hist_of_all_channel = [i * weights[cropIdx] for i in hist_of_all_channel]
            hist_features.append(np.stack(hist_of_all_channel, axis=0))

        hist_features = np.stack(hist_features, axis=0)
        if return_crops:
            return hist_features, all_crops
        else:
            return hist_features

    @staticmethod
    def get_lbp_imgPIL(imgPIL, R=1, P=None):
        local_binary_pattern = _get_module("skimage.feature.local_binary_pattern")
        if P is None:
            P = 8 * R
        return Image.fromarray(np.array(local_binary_pattern(np.array(imgPIL.convert("L")), P=P, R=R)))

    class Hist:
        def __init__(self, row=None, col=None, bins=None, p_weight=None):
            self.row = 4 if row is None else row
            self.col = 4 if col is None else col
            # 如果p_weight, row, col都没有提供，就使用默认的全套
            # IMPORTANT 不建议这里对不同区域的直方图直接乘上大于1的权重，会导致直方图某个bins里溢出（超过1）
            self.partial_weight = np.array([[1, 1, 1, 1],
                                            [1, 1, 1, 1],
                                            [1, 1, 1, 1],
                                            [1, 1, 1, 1]]) if all(i is None for i in [p_weight,row,col]) else p_weight
            assert self.partial_weight.shape == (self.row, self.col), "权重矩阵必须和分块的row、col先匹配"
            self.bins = [32, 64, 32] if bins is None else bins

        def get_hist(self, imgPIL, return_crops=False, nested=False):
            """
            返回形如： shape=(row*col, 3, bins)
            ( (part1_r, part1_g, part1_b),
              (part2_r, part2_g, part2_b)
              ... )
            """
            res = StandardCV.get_hist(imgPIL,row=self.row,col=self.col,bins=self.bins,
                                                      weights=self.partial_weight.flatten(), return_crops=return_crops)
            if nested:
                if return_crops:
                    return res[0].reshape(self.row * self.col, -1), res[1]
                else:
                    return res.reshape(self.row*self.col, -1)
            else:
                return res

        def imgPIL2vec(self, imgPIL):
            """
            返回colorHistogram的 concatenate / 展平 的向量
            具体来说，各子区域展平，区域内三个通道展平 (part1_r : part1_g : part1_b : part2_r : part2_g ...)
            """
            return self.get_hist(imgPIL, return_crops=False).flatten()

        def imgPIL2vec_nested(self, imgPIL):
            # 每个子图区域里，三通道拼接到一起，shape是 (row*col, 3, bins) --> (row*col, 3*bins)
            return self.get_hist(imgPIL, return_crops=False, nested=True)

        def get_lbp_hist(self, imgPIL, cutShape=None, bins=None, weight=None, R=1, P=None, return_crops=False, nested=False):
            imgPIL_lbp = StandardCV.get_lbp_imgPIL(imgPIL, R, P)
            if cutShape is None:
                cutShape = (self.row, self.col)
            if bins is None:
                bins = [32]
            if weight is None:
                weight = self.partial_weight
            res = StandardCV.get_hist(imgPIL_lbp, row=cutShape[0], col=cutShape[1], bins=bins,
                                              weights=weight.flatten(), return_crops=return_crops)
            if nested:
                if return_crops:
                    return res[0].reshape(cutShape[0]*cutShape[1], -1), res[1]
                else:
                    return res[0].reshape(cutShape[0]*cutShape[1], -1)
            else:
                return res

        def imgPILLBP2vec(self, imgPIL, cutShape=None, bins=None, weight=None, R=1, P=None):
            return self.get_lbp_hist(imgPIL, cutShape, bins, weight, R, P, return_crops=False).flatten()

        def imgPILLBP2vec_nested(self, imgPIL, cutShape=None, bins=None, weight=None, R=1, P=None):
            # 每个子图区域里，三通道拼接到一起，shape是 (row*col, bins)
            return self.get_lbp_hist(imgPIL, cutShape, bins, weight, R, P, return_crops=False, nested=True)

class Scripts:
    GRAY_MODE = ['F','L']
    default_color_map = {
        "RGB": ['r', 'g', 'b'],
        "YCbCr": ['r', 'black', 'b'],
        "F": ['black'],
        "L": ['black'],
    }
    @staticmethod
    def cos_sim(arr1, arr2):
        return np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))

    @staticmethod
    def cos_sim_nested(hist1, hist2, weight=None):
        if weight is None:
            weight = [1] * hist1.shape[0]
        sim_list = [Scripts.cos_sim(hist1[i, :], hist2[i, :]) * weight[i] for i in range(hist1.shape[0])]
        return sum(sim_list) / len(sim_list)

    @staticmethod
    def plot_hist_crops(hist_inp, crops_inp, color_map=None, dont_show=None):
        row, col = crops_inp.shape
        crops_inp_img = np.concatenate(crops_inp, axis=0)
        mode = crops_inp_img[0].mode
        if mode == 'RGB':
            print("not recommend RGB for similarity, try YCbCr")
        is_gray = mode in Scripts.GRAY_MODE

        if dont_show is None:
            dont_show = []

        if color_map is None:
            try:
                color_map = Scripts.default_color_map[mode]
            except Exception:
                print("can't auto define color_map. must feed this param")

        fig = plt.figure(figsize=(10, 10))
        fig.set_tight_layout(True)
        for idx, (crop_hist, crop_img) in enumerate(zip(hist_inp, crops_inp_img)):
            fig.add_subplot(row, 2 * col, 2 * idx + 1)
            plt.title("hist_{}".format(mode))
            for idx_, i in enumerate(crop_hist):
                if color_map[idx_] in dont_show:
                    continue
                plt.plot(i, color=color_map[idx_])
                plt.xlim([0, crop_hist.shape[-1]])  # 手工设置x轴的最大值（默认用数据里的最大值）
                plt.ylim([0, 1])
            fig.add_subplot(row, 2 * col, 2 * idx + 2)
            plt.title("RGB_{}".format(idx))
            plt.axis('off')
            plt.tight_layout(h_pad=1, w_pad=0.2)
            if is_gray:
                plt.imshow(crop_img, cmap='gray')
            else:
                plt.imshow(crop_img.convert("RGB"), cmap='viridis')
        return fig

if __name__ == '__main__':
    test_url = "http://www.kedo.gov.cn/upload/resources/image/2017/04/24/150703.png"
    test_url2 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThwIfzyp-Rv5zYM0fwPmoM5k1f9eW3ETYuPcL8j2I0TuG0tdb5&s"
    from zac_pyutils import CVUtils
    test_img = CVUtils.Load.image_by_pil_from(test_url).convert("YCbCr")
    test_img2 = CVUtils.Load.image_by_pil_from(test_url2).convert("YCbCr")
    test_img.show()
    test_img2.show()

    def test_case0():
        print(">>> 验证切图是否正常")
        cropsPIL_matrix = StandardCV.custom_cut_to_matrix(test_img, row=4, col=4)
        for i in range(cropsPIL_matrix.shape[0]):
            for j in range(cropsPIL_matrix.shape[1]):
                plt.subplot(4, 4, i * 4 + (j + 1))
                plt.imshow(cropsPIL_matrix[i,j])

        print(">>> 展示切割后各部分的直方图")
        transformer = StandardCV.Hist()
        hist, crops=transformer.get_hist(test_img, return_crops=True)
        Scripts.plot_hist_crops(hist, crops)
        plt.show()

    def test_case1():
        print(">>> 验证StandardCV的直方图方式相似度量")
        transformer = StandardCV.Hist()
        print("多个子区域的直方图拼接成一个向量计算cos_sim: {:.4f}".format(Scripts.cos_sim(transformer.imgPIL2vec(test_img), transformer.imgPIL2vec(test_img2))))
        print("多个子区域的直方图独立计算cos_sim然后取平均: {:.4f}".format(Scripts.cos_sim_nested(transformer.imgPIL2vec_nested(test_img), transformer.imgPIL2vec_nested(test_img2))))

    def test_case2():
        print(">>> 验证NN相似向量")
        transformer = VectorFromNN()
        vec1 = transformer.imgPIL2vec(test_img)
        vec2 = transformer.imgPIL2vec(test_img2)
        from ImageFeature import Hist
        print("NN取最后一层计算cos_sim: {:.4f}".format(Hist.General.cos_sim(vec1, vec2)))
        print(f"向量一: {vec1.shape} {type(vec1)}\n", vec1[:10])
        print(f"向量二: {vec2.shape} {type(vec2)}\n", vec2[:10])

    test_case1()
    test_case2()

