# author: zac
# create-time: 2019-11-04 10:04
# usage: -
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import local_binary_pattern
import skimage

class PILImage:
    GRAY_MODE=['F','L']
    @staticmethod
    def get_vector(img_inp, lbp_weight=1.0, color_weight=1.0, bins_lbp=None, bins_color=None):
        """
        返回concatenate的展平的colorHistogram和LBPH向量
        :param img_inp:
        :param lbp_weight:
        :param color_weight:
        :param bins_lbp:
        :param bins_color:
        :return:
        """
        if bins_lbp is None:
            bins_lbp = [32]
        if bins_color is None:
            bins_color = [32,64,32]
        img_lbp = PILImage.get_img_lbp(img_inp, R=1)
        weights = np.array([[1, 1, 1, 1],
                            [1, 2, 2, 1],
                            [1, 2, 2, 1],
                            [1, 1, 1, 1]])
        hist_rgb, crops_rgb = PILImage.get_hist(img_inp, row=4, col=4, bins=bins_color, weights=weights.flatten(), return_crops=True)
        hist_lbp, crops_lbp = PILImage.get_hist(img_lbp, row=4, col=4, bins=bins_lbp, weights=weights.flatten(), return_crops=True)
        return np.concatenate([hist_rgb.flatten() * color_weight, hist_lbp.flatten() * lbp_weight])

    @staticmethod
    def get_vector_nested(img_inp, lbp_weight=1.0, color_weight=1.0, bins_lbp=None, bins_color=None):
        if bins_lbp is None:
            bins_lbp = [32]
        if bins_color is None:
            bins_color = [32,64,32]
        img_lbp = PILImage.get_img_lbp(img_inp, R=1)
        hist_rgb, crops_rgb = PILImage.get_hist(img_inp, row=4, col=4, bins=bins_color, return_crops=True)
        hist_lbp, crops_lbp = PILImage.get_hist(img_lbp, row=4, col=4, bins=bins_lbp, return_crops=True)
        hist_rgb = np.array([h.flatten() for h in hist_rgb])
        hist_lbp = np.array([h.flatten() for h in hist_lbp])
        return np.concatenate([hist_rgb * color_weight, hist_lbp * lbp_weight], axis=1)

    @staticmethod
    def cut_to_custom_part(img_inp, row, col):
        # 0 1 2
        # 3 4 5
        # 6 7 8
        #    col0       col1       col2       col3
        # (  0,  0)  (1/3,  0)  (2/3,  0)  (3/3,  0)   row0
        # (  0,1/3)  (1/3,1/3)  (2/3,1/3)  (3/3,1/3)   row1
        # (  0,2/3)  (1/3,2/3)  (2/3,2/3)  (3/3,2/3)   row2
        # (  0,3/3)  (1/3,3/3)  (2/3,3/3)  (3/3,3/3)   row3
        crop_matrix = []
        base_w, base_h = img_inp.size[0] // col, img_inp.size[1] // row
        for r in range(row):
            crop_one_row = []
            for c in range(col):
                anchor_lt = (base_h*c, base_w*r)  # 左上角的anchor
                anchor_rb = (base_h*(c+1), base_w*(r+1))  # 右下角的anchor
                crop_one_row.append(img_inp.crop((anchor_lt[0], anchor_lt[1], anchor_rb[0], anchor_rb[1])))
            crop_matrix.append(crop_one_row)
        return np.array(crop_matrix,dtype='object')


    @staticmethod
    def get_hist(img_inp, row=1, col=1, bins=None, weights=None, return_crops=False):
        """
        根据 row,col 会切分图像，最后返回时 all_crops 的size是(row, col, img_w, img_h, img_channel)
        各区域单独计算hist然后拼接到一起
        :param img_inp:
        :param row:
        :param col:
        :param bins: 设定各个通道的bins个数，默认[32]*3。如果不是一样大的会padding到和max一样大
                     例如R通道4个bins: [16,16,16,32]，G通道2个bins：[32,48]，G会padding为[32,48,0,0]
        :param weights: 各个区域hist权重，默认都为1
        :param return_crops:
        :return:
        """
        if bins is None:
            bins=[32]*3
        if weights is None:
            weights = [1.0] * row * col
        is_gray = len(np.array(img_inp).shape) == 2
        # 切块
        all_crops = PILImage.cut_to_custom_part(img_inp, row, col)  # come with row x col
        all_crops_flatten = np.concatenate(all_crops, axis=0)  # flattent to (16, 128, 128, 3)

        hist_features = []
        for crop_idx, crop_img in enumerate(all_crops_flatten):
            crop_arr = np.array(crop_img)
            # np.array(crop).shape[-1] 是取通道数
            # np.histogram 返回结果两个， hist, bins_edge —— 直方图和bin的方位
            if is_gray:
                # 灰度图的hist为了方便后面统一使用，也在外面包上一个维度
                hist_of_all_channel = [np.histogram(crop_arr, bins=bins[0], range=[0, 256])[0]]
            else:
                hist_of_all_channel = [np.histogram(crop_arr[:, :, c], bins=bins[c], range=[0, 256])[0] for c in range(crop_arr.shape[-1])]
            # normalize
            hist_of_all_channel = [hist/sum(hist) for hist in hist_of_all_channel]
            # 不同通道bins可能有不同，末尾padding到相同
            # 例如R通道4个bins: [16,16,16,32]，G通道2个bins：[32,48]，G会padding为[32,48,0,0]
            max_bin = max(bins)
            hist_of_all_channel = [np.pad(hist, (0,max_bin-bins[idx]),'constant') for idx,hist in enumerate(hist_of_all_channel)]
            # 此区域三通道的hist都乘上权重
            hist_of_all_channel = [i * weights[crop_idx] for i in hist_of_all_channel]
            hist_features.append(np.stack(hist_of_all_channel, axis=0))

        hist_features = np.stack(hist_features, axis=0)
        if return_crops:
            return hist_features, all_crops
        else:
            return hist_features


    @staticmethod
    def get_img_lbp(img_inp, R=1, P=None):
        if P is None:
            P = 8*R
        return Image.fromarray(np.array(local_binary_pattern(np.array(img_inp.convert("L")), P=P, R=R)))


class General:
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
    def cos_sim_nested(hist1,hist2, weight=None):
        if weight is None:
            weight = [1] * hist1.shape[0]
        sim_list=[General.cos_sim(hist1[i,:], hist2[i,:]) * weight[i] for i in range(hist1.shape[0])]
        return sum(sim_list) / len(sim_list)

    @staticmethod
    def plot_hist_crops(hist_inp, crops_inp, color_map=None, dont_show=None):
        row, col = crops_inp.shape
        crops_inp_img = np.concatenate(crops_inp, axis=0)
        mode = crops_inp_img[0].mode
        if mode == 'RGB':
            print("not recommend RGB for similarity, try YCbCr")
        is_gray = mode in PILImage.GRAY_MODE


        if dont_show is None:
            dont_show = []

        if color_map is None:
            try:
                color_map = General.default_color_map[mode]
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
