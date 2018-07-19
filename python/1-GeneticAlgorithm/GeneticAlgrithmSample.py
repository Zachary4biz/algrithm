# -*- coding: UTF-8 -*-
import random
from PIL import Image, ImageDraw
import numpy as np
import time
from progressbar import *
import pickle
import os
import sys

# python 2.7
# 思路参考:http://songshuhui.net/archives/10462
# abspath 是执行脚本的路径, sys.argv[0]是执行时给的路径。举例如下:
# 在 ~/ 目录下执行 ` python local_path/dir/ur_py_script.py ` abspath是 "/Users/zac", sys.argv[0]的dirname是 "local_path/dir"
# 如果cd到当前目录中再执行 ` python ur_py_script.py ` abspath是 "local_path/dir", sys.argv[0]就是ur_py_script.py,没有dirname
basePath = os.path.abspath('.')
pyFilePath = os.path.dirname(sys.argv[0])
use_path = ""
if pyFilePath=="":
    # 如cd到py脚本的根目录直接执行 python xxx.py
    use_path = basePath
elif basePath in pyFilePath:
    # 如cd到~目录执行 python /User/zac/Algrithm/xxx.py
    use_path = pyFilePath
else:
    # 如cd到py脚本的根目录的上一级目录执行 python script_dir/xxx.py
    use_path = basePath + "/"+ pyFilePath
print ("use_path is "+ use_path)
print ("pyFilePath is " + pyFilePath, "basePath is "+basePath)
img_target = Image.open(use_path+"/genetic_img/firefox.jpeg")
img_savePath = use_path+"/genetic_img"
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
        self.loss = 0.0

if __name__ == '__main__':
    ###
    # 淘汰率设为1500的时候,每轮迭代飞快,本来一轮 4s,现在 1s 十轮,但是持续几千轮每轮的最优都没有变化
    ###
    print("=======>  begin")
    # 初始化100000个
    herd_size = 8000
    # 初始100个基因
    gene_cnt = 200
    # 迭代80w代
    r_generation = 300000
    # 每次迭代淘汰 1000个,随着迭代进行,每进行 2000轮淘汰个数减少100个,淘汰个数最少 200
    weed_out = 400
    weed_update_lock = False
    epoch_weed = 3000
    min_weed_out = 200
    # 变异率初期小,后期迭代进行变异率增加,最多0.6
    varation_init_rate_global = 0.1
    varation_update_lock = True
    epoch_varation = 3000
    varation_update_rate = 0.05
    max_varation_rate_global = 0.5
    # 生成一个子代的方法
    def copulation(scollop, scollop_another, varation_rate, max_varation_rate = 0.9):
        new_position = random.sample(scollop.positions, int(scollop.gene_count * 0.5)) + random.sample(scollop_another.positions,int(scollop_another.gene_count * 0.5))
        new_scollop = Scollop(gene_count=len(new_position), size=scollop.size, positions=new_position)
        img = Image.new('RGB', scollop.size, color=(255, 255, 255))
        drw = ImageDraw.Draw(img, 'RGBA')
        varation_rate_t = max_varation_rate if varation_rate>= max_varation_rate else varation_rate
        for p in new_position:
            if random.random()<=varation_rate_t:
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
    # 淘汰方式
    def select_func(generation_sample, topN):
        # 直接进行末尾淘汰
        selected = sorted(generation_sample, key=lambda x: x[1])[:topN]
        return selected
    ####
    # 生成下一代
    # 两两交配生成大量子代
    # 从子代中选择最优的weed_out个填充到下一迭代过程
    def sub_generation(generation_sample, varation_rate,weed_cnt):
        born = list()
        for _ in range(int(len(generation_sample)*0.5)):
            s1,s2=map(lambda x:x[0], random.sample(generation_sample,2))
            # 迭代生成新的Scollop
            s_new = copulation(s1,s2,varation_rate,max_varation_rate=max_varation_rate_global)
            # 计算loss
            loss = calculate_loss(s_new.img, img_target)
            # 合并
            born.append((s_new,loss))
        # 子代取最优的n个
        topN = select_func(born,weed_cnt)
        return topN
    ####
    # 生成下一代
    # 最优n个生成一半子代,随机n个生成另一半子代
    def sub_generation_v2(generation_sample,varation_rate,weed_cnt):
        # 挑选 topN 个互相杂交, 再全局随机挑选 2n 个互相杂交,最后得到n个子代
        topN_cop = select_func(generation_sample,weed_cnt)
        top_born = list()
        random_cop = random.sample(generation_sample,2*weed_cnt)
        random_born = list()
        for _ in range(int(len(topN_cop)*0.5)):
            s1,s2=map(lambda x:x[0], random.sample(topN_cop,2))
            # 迭代生成新的Scollop
            s_new = copulation(s1,s2,varation_rate,max_varation_rate=max_varation_rate_global)
            # 计算loss
            loss = calculate_loss(s_new.img, img_target)
            # 合并
            top_born.append((s_new,loss))
        for _ in range(int(len(random_cop)*0.5)):
            s1,s2=map(lambda x:x[0], random.sample(random_cop,2))
            # 迭代生成新的Scollop
            s_new = copulation(s1,s2,varation_rate,max_varation_rate=max_varation_rate_global)
            # 计算loss
            loss = calculate_loss(s_new.img, img_target)
            # 合并
            random_born.append((s_new,loss))
        return select_func(top_born + random_born,weed_cnt)

    ####
    # 生成下一代:锦标赛方式
    # 每次随机杂交,多轮杂交后每轮取最优的一个(或几个),直到获得weed_out个新的子代
    def sub_generation_v3(generation_sample,varation_rate,weed_cnt):
        result = list()
        random_cop = random.sample(generation_sample,2*weed_cnt)
        topN_of_each_loop = 10
        for iter in range(int(math.ceil(weed_cnt/topN_of_each_loop))):
            random_born = list()
            for __ in range(int(weed_cnt)):
                s1,s2=map(lambda x:x[0], random.sample(random_cop,2))
                # 迭代生成新的Scollop
                s_new = copulation(s1,s2,varation_rate,max_varation_rate=max_varation_rate_global)
                # 计算loss
                loss = calculate_loss(s_new.img, img_target)
                # 合并
                random_born.append((s_new,loss))
            result.extend(select_func(random_born,topN_of_each_loop))
        return select_func(result,weed_cnt)

    # 动态更新变异率(因为迭代到后来,几千轮都没有更好的最优解出现,所以考虑加大变异率)
    def dynamic_varation(v, cnt):
        if cnt >= 1000 and varation_update_lock:
            # 已经超过1k轮最优解相同的情况下:
            # 每增加1k轮相同,会导致v增加10*update_rate(0.05)
            # 0.05 -> 0.0505 -> 0.051 -> 0.0515 -> 0.052 ... -> 0.1 (cnt=1100) -> ...
            v = v+1.0*(cnt-1000)/100*varation_update_rate if v<=max_varation_rate_global else max_varation_rate_global
        else:
            # 如果未开启自适应变异 || cnt重置为1了(表示有新的最优解了) => 变异率重置
            v = varation_init_rate_global
        return v

    # 持续 n 轮没有更优解的时候变异率直接增加到0.8进行一轮迭代然后重置回去,目的是让变异后的基因有充分迭代杂交的机会
    # 具体实现就是通过cnt获得"当前已经连续cnt轮没有更优解",每当cnt%1000=0的时候就把变异率拔高到0.8,其他时候重置
    def dynamic_varation_v2(v,cnt):
        if cnt % 500==0 and varation_update_lock:
            v = 0.6 if v<=max_varation_rate_global else max_varation_rate_global
        else :
            v = varation_init_rate_global
        return v
    # ---------------------------------------------------------------------------------------------------------
    # 初始化种群
    print ("inital ancestor....")
    ancestor_sample = list()
    pbar_ancestor = ProgressBar(widgets=[Percentage(),Bar(),' ', Timer(),], maxval=herd_size).start()
    for i in range(herd_size):
        s = Scollop(gene_cnt, img_target.size)
        s.ancestor()
        # 计算loss
        ancestor_loss = calculate_loss(s.img, img_target)
        # 合并
        ancestor_sample.append((s, ancestor_loss))
        pbar_ancestor.update(i+1)
    pbar_ancestor.finish()
    print ("ancestor done!")
    ####
    # 这里暂时不用,发现需要的空间太大了, 150x135的图,10W的种群需要6.6G空间
    # 把s保存起来免得以后一直需要初始化种群
    # f=open(savePath,"wb+")
    # pickle.dump(all_sample,f,protocol=pickle.HIGHEST_PROTOCOL)
    # f.close()
    #
    # # 加载这一千个祖宗
    # f_read=open(savePath,"rb")
    # all_sample_read = pickle.load(f_read)
    ####

    ancestor_sample_read=ancestor_sample
    sample_selected = list()
    # 记录一下有多少代的最优解相同,100代大概是9min
    cnt_of_same_best = 1
    # 初始化变异率
    varation_input = varation_init_rate_global
    print ("begin generating..")
    for generation in range(r_generation):
        # 自然选择
        sample_selected = select_func(ancestor_sample_read, herd_size - weed_out)
        # 每100代展示一次最优结果
        if generation % 1 == 0:
            print ("generation: %s, best loss: %s, varation: %s, time: %s" % (generation, sample_selected[0][1], varation_input,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        # 每500代保存一次结果
        if (generation>=1000 and generation % 100 == 0) or (generation<=100 and generation %5 == 0) or ( 100< generation <= 1000 and generation % 10 == 0):
            sample_selected[0][0].img.save("%s/%s_%s_%s.jpg" % (img_savePath, generation, sample_selected[0][1],varation_input))
        # 变异率
        varation_input = dynamic_varation_v2(v=varation_input,cnt=cnt_of_same_best)
        # 生成下一代
        top_of_new_born = sub_generation_v3(sample_selected, varation_input, weed_out)
        # 子代和父代合并
        ancestor_sample_read = sample_selected + top_of_new_born
        # 如果子代中没有产生更优解,说明这一轮后的最优解没有变化
        if top_of_new_born[0][1] <= sample_selected[0][1]:
            cnt_of_same_best +=1
        else:
            cnt_of_same_best = 1




        #
        # # 剩余的900个两两组合
        # new_born = list()
        # for _ in range(weed_out):
        #     # 第一个
        #     s1,s2=map(lambda x:x[0], random.sample(all_sample_selected,2))
        #     # 变异率随着迭代进行会增加
        #     if generation % epoch_varation == 0 and varation_rate < max_varation_rate and varation_update_lock:
        #         varation_rate =+ 0.05
        #         print ("varation_rate now is %s" % varation_rate)
        #     # 迭代生成新的Scollop
        #     s_new = copulation(s1,s2)
        #     # 计算loss
        #     loss = calculate_loss(s_new.img, img_target)
        #     # 合并
        #     new_born.append((s_new,loss))
        # all_sample_read = all_sample_selected + new_born
        # time_end = time.time()

