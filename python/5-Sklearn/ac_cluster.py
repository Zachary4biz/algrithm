# encoding=utf-8
import pandas as pd
import codecs
import numpy as np
import time
import re
import math
from sklearn.cluster import DBSCAN

pd.set_option('display.height',1000)
pd.set_option('display.width',800)
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',12)


# 特征清理
def clean_data(input_df):
    # 去掉列
    id_to_drop=['version_name_s','product_id_l']
    single_value_drop = ['cpu_frequence_d', 'is_pad_l',"last_connect_time_l", "last_response_time_l", 'support_camera_l', 'support_nfc_l']
    duplicates_col_drop=['request_time_l', 'server_time_l']
    too_much_null_drop = ['last_connected_ip_s', 'last_transfer_rate_l', 'sim_country_s']
    # meaningless_drop = ['referrer_s']
    cols_to_drop = id_to_drop+single_value_drop+duplicates_col_drop+too_much_null_drop
    input_df.drop(columns=cols_to_drop, inplace=True)
    # 时间戳解析
    for fieldName in ['event_time_l']:
        input_df[fieldName]=input_df[fieldName].apply(lambda dt: time.strftime("%Y-%m-%d %H:%M", time.localtime(int(dt) / 1000)))
    # 特殊值处理
    input_df['package_info']=input_df.apply(lambda row: row['package_name_s'] + " && " + row['version_code_l'], axis=1)
    input_df.drop(columns=['package_name_s','version_code_l'], inplace=True)
    input_df['is_roaming_b']=input_df['is_roaming_b'].apply(lambda x: x if x == "ture" else "")
    input_df['network_type_l']=input_df['network_type_l'].apply(lambda x: x if x != "0" else "")
    input_df['channel_id_s'] = input_df['channel_id_s'].apply(lambda x: x if x not in ['100000','google-play'] else "")
    input_df['os_version_s'] = input_df['os_version_s'].apply(lambda x: re.findall(pattern="[0-9]\\.[0-9]\\.[0-9]", string=x)[0] if len(re.findall(pattern="[0-9]\\.[0-9]\\.[0-9]", string=x)) > 0 else "")
    # input_df['channel_tag_s'] = input_df['channel_tag_s'].apply(lambda x: x if x !='GP' else "")
    return input_df

def feature_transform(input_df):
    # from sklearn.preprocessing import LabelEncoder
    # from sklearn.preprocessing import OneHotEncoder
    # input_df['country_s'] = LabelEncoder().fit_transform(y=input_df['country_s'])
    input_df['client_ip_init_s'] = input_df['client_ip_s'].apply(lambda x: ".".join(x.split(".")[:3]))
    return input_df

def useless_feature_drop(input_df):
    # todo: 这里面有些是因为处理后的特征全都是空串,如果用jaccard相似到可以把这种空串筛出不计入计算,但是欧氏距离,这种空串的特征大部分都相同了
    # 注: 其实可以把空串替换为 NaN,然后直接pd.get_dummies ,会自动过滤掉 特征=NaN
    cols_to_drop = ['channel_tag_s','continent_s','country_s',
                    'language_s','locale_s','province_s','referrer_s_utm_source']
    input_df.drop(columns=cols_to_drop, inplace=True)
    return input_df

def main():
    # 读入原始数据
    p='/Users/zac/Desktop/tlv_register/part-00001-91f2e808-d9d3-4a39-952a-768f979d4ad5.csv'
    # p = '/data/houcunyue/zhoutong/data/tlv_register.csv'
    with codecs.open(p,encoding="utf-8",mode='r') as f:
        content=f.readlines()
        info = list(map(lambda x:x.strip().split(u"\u0394"), content))

    df = pd.DataFrame(np.array(info[1:]),columns=info[0])
    input_df = clean_data(df)
    input_df = feature_transform(input_df)
    input_df = useless_feature_drop(input_df)
    input_df = input_df.replace("",np.nan).dropna(axis='columns',how='all')
    ########
    # IMPORTANT: get_dummies 会自动过滤不编码NaN
    # 但是,似乎如果一列全为NaN,dummy_df会编码出一列 is_roaming_b ,值既不为1也不为0,全为NaN
    # 所以需要一个 .dropna(axis='columns',how='all').columns ,把全NaN的丢掉
    #######
    dummy_df = pd.get_dummies(input_df,prefix_sep="=")
    dummy_df = dummy_df.sample(n=600)

    # 把 ip、event_time_l、权重增大
    # DBSCAN原生的方法只支持样本的权重,不支持特征的权重,得从metric下手或者修改dummy_df的值
    # 遍历 dummy_df,对于带有指定字符的列,如果值是1就改为 1.5 或 2,即增加权重
    all_field = list(dummy_df.columns)
    # weight_w1_filed = list(filter(lambda x: 'client_ip_init_s' in x,all_field))
    weight_w2_filed = list(filter(lambda x: 'client_ip_s' in x or "event_time_l" in x,all_field))
    # dummy_df[weight_w1_filed] = dummy_df[weight_w1_filed].replace(1,1.5)
    dummy_df[weight_w2_filed] = dummy_df[weight_w2_filed].replace(1,2)

    from sklearn.metrics.pairwise import euclidean_distances
    euclidean_distances(X=dummy_df)
    ######
    # 使用欧氏距离
    # one-hot编码后特征都是 1, 共23个特征
    # 所以如果23个特征不一样,欧氏距离就是 math.pow(46,0.5)=6.7823
    # 如果控制 4个特征一样,欧氏距离就是 math.pow(46-2*4,0.5) = 6.1644
    # 如果控制15个特征一样,欧氏距离就是 math.pow(46-2*15,0.5) = 4.0
    # 如果有一个权重是2的特征不一样,欧氏距离至少是 (2^2+2^2)^0.5 math.pow(8,0.5)
    # 注: 以上只在两个样本特征总数一样的情况下,使用了NaN替换空串特征后,dummy编码不会编码NaN,这样两个样本的特征总数可能就不一样了
    total_feature_cnt = 12
    same_feature_cnt = 7
    eps = math.pow(2*(12-6),0.5)
    cluster_result = DBSCAN(eps=4.5, min_samples=15).fit_predict(X=dummy_df)
    ######
    # 尝试使用jaccard度量
    jaccard_sim = 0.3
    jaccard_diff = 1- jaccard_sim
    # cluster_result = DBSCAN(eps=1-0.2, min_samples=15, metric='jaccard').fit_predict(X=sample_df)

    print("各个类别的样本个数统计")
    print(list(zip(np.unique(cluster_result[cluster_result!=-1]),np.bincount(cluster_result[cluster_result!=-1]))))
    print("各个类别的idx")
    for i in np.unique(cluster_result[cluster_result!=-1]):
        print("====> 类别 %s" % i)
        idx_list = np.where(cluster_result==i)[0]
        for j in idx_list:
            user_info = dummy_df.iloc[j][dummy_df.iloc[j]!=0]
            info_str = ",".join(list(pd.DataFrame(user_info).T.columns))
            print(info_str)

    print(cluster_result.max)
    # cluster_df

def get_dummy_reverse(df):

    pass


if __name__ == '__main__':
    pass



# 来自scala生成的数据: spark.read.parquet("/user/hive/warehouse/dw_events.db/dw_events_tlv_hour/dt=2018-06-0[1-3]/pn=*/hour=*/et=filterregister").distinct.repartition(10).write.option("header","true").option("delimiter","\u0394").mode("overwrite").csv("tlv_register")
