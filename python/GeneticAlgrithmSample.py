# -*- coding: UTF-8 -*-
import random
from PIL import Image, ImageDraw
import numpy as np
import time
from progressbar import *
import pickle
import os
import sys

# 思路参考:http://songshuhui.net/archives/10462
basePath = os.path.abspath('.')
pyFilePath = os.path.dirname(sys.argv[0])
img_target = Image.open(pyFilePath+"/genetic_img/timg.jpeg")
img_savePath = pyFilePath+"/genetic_img"
# savePath = "/Users/zac/Downloads/Ancestor.pk"

class Scollop(object):
    @staticmethod
    def generate_triangle(img, width, height):
        point_a = (random.randint(0, width), random.randint(0, height))
        point_b = (random.randint(0, width), random.randint(0, height))
        point_c = (random.randint(0, width), random.randint(0, height))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        drw = ImageDraw.Draw(img, 'RGBA')
        drw.polygon([point_a, point_b, point_c], color)
        return point_a, point_b, point_c, color

    def ancestor(self):
        for _ in range(self.gene_count):
            self.positions.append(self.generate_triangle(self.img, self.size[0], self.size[1]))

    def __init__(self, gene_count, size, positions=list()):
        self.gene_count = gene_count
        self.positions = positions
        self.size = size
        self.img = Image.new('RGB', self.size, color=(255, 255, 255))

if __name__ == '__main__':
    ###
    # 淘汰率设为1500的时候,每轮迭代飞快,本来一轮 4s,现在 1s 十轮,但是持续几千轮每轮的最优都没有变化
    ###
    print("begin")
    # 初始化100000个
    herd_size = 5000
    # 每次迭代淘汰 1000个,随着迭代进行,每进行 2000轮淘汰个数减少100个,淘汰个数最少 200
    weed_out = 200
    weed_update_lock = False
    epoch_weed = 3000
    min_weed_out = 50
    # 变异率初期小,后期迭代进行变异率增加,最多0.6
    varation_rate = 0.3
    varation_update_lock = False
    epoch_varation = 3000
    max_varation_rate = 0.5
    # 迭代方式
    def copulation(scollop, scollop_another):
            new_position = random.sample(scollop.positions, int(scollop.gene_count * 0.5)) + random.sample(scollop_another.positions,
                                                                                                           int(scollop_another.gene_count * 0.5))
            new_scollop = Scollop(gene_count=len(new_position), size=scollop.size, positions=new_position)
            img = Image.new('RGB', scollop.size, color=(255, 255, 255))
            drw = ImageDraw.Draw(img, 'RGBA')
            for p in new_position:
                if random.random()<=varation_rate:
                    new_scollop.generate_triangle(img,scollop.size[0],scollop.size[1])
                else:
                    drw.polygon([p[0], p[1], p[2]], p[3])
            new_scollop.img = img
            return new_scollop
    # loss损失函数
    def calculate_loss(image1,image2):
        img_1 = np.asarray(image1, dtype="int32")
        img_2 = np.asarray(image2, dtype="int32")
        # loss_result = np.count_nonzero(img_1 - img_2)
        # loss_result = np.dot(img_1.flatten(),img_2.flatten())/(np.sqrt(sum(img_1.flatten()*img_1.flatten()))*np.sqrt(sum(img_2.flatten()*img_2.flatten())))
        # loss_result = sum(map(sum, np.nonzero(img_1 - img_2)))
        loss_result = 1.0*np.abs(img_1-img_2).sum()/np.abs(img_1-img_2).size
        return loss_result

    all_sample = list()
    pbar_ancestor = ProgressBar(widgets=[Percentage(),Bar(),' ', Timer(),], maxval=herd_size).start()
    for i in range(herd_size):
        s = Scollop(100, img_target.size)
        s.ancestor()
        # 计算loss
        loss = calculate_loss(s.img, img_target)
        # 合并
        all_sample.append((s, loss))
        pbar_ancestor.update(i+1)
    pbar_ancestor.finish()

    print ("ancestor done!")
    # 把s保存起来免得以后一直需要初始化种群
    # 150x135的图,10W的种群需要6.6G空间
    # f=open(savePath,"wb+")
    # pickle.dump(all_sample,f,protocol=pickle.HIGHEST_PROTOCOL)
    # f.close()
    #
    # # 加载这一千个祖宗
    # f_read=open(savePath,"rb")
    # all_sample_read = pickle.load(f_read)
    all_sample_read=all_sample
    # 迭代80w代
    r_generation = 300000
    # pbar_generation = ProgressBar(widgets=[Percentage(),Bar()], maxval=r_generation).start()
    for generation in range(r_generation):
        time_start = time.time()
        # 淘汰loss最高的100个
        all_sample_selected = sorted(all_sample_read, key=lambda x: x[1])[:(herd_size - weed_out)]
        # 淘汰率随着迭代进行会减少
        if generation % epoch_weed == 0 and weed_out > min_weed_out and weed_update_lock:
            weed_out =- 10
            print ("weed_out at %s, generation: %s, epoch_weed: %s , %s" % (weed_out,generation,epoch_weed,generation % epoch_weed))
        # 每100代展示一次最优结果
        if generation % 100 == 0:
            print ("generation: %s, best loss: %s" %(generation,all_sample_selected[0][1]))
        if generation % 100 == 0:
            all_sample_selected[0][0].img.save("%s/%s_%s.jpg" % (img_savePath,generation,all_sample_selected[0][1]))
        # 剩余的900个两两组合
        new_born = list()
        for _ in range(weed_out):
            # 第一个
            s1,s2=map(lambda x:x[0], random.sample(all_sample_selected,2))
            # 变异率随着迭代进行会增加
            if generation % epoch_varation == 0 and varation_rate < max_varation_rate and varation_update_lock:
                varation_rate =+ 0.05
                print ("varation_rate now is %s" % varation_rate)
            # 迭代生成新的Scollop
            s_new = copulation(s1,s2)
            # 计算loss
            loss = calculate_loss(s_new.img, img_target)
            # 合并
            new_born.append((s_new,loss))
        all_sample_read = all_sample_selected + new_born
        time_end = time.time()
    #     pbar_generation.update(generation+1)
    # pbar_generation.finish()
