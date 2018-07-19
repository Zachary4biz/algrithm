# encoding=utf-8
import pandas as pd
import sklearn
import sys
import codecs
import numpy as np
import time
import re

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
        input_df[fieldName]=pd.to_datetime(input_df[fieldName].apply(lambda dt: time.strftime("%Y-%m-%d %H:%M", time.localtime(int(dt) / 1000))))
    # 特殊值处理
    input_df['package_info']=input_df.apply(lambda row: row['package_name_s'] + " && " + row['version_code_l'], axis=1)
    input_df.drop(columns=['package_name_s','version_code_l'], inplace=True)
    input_df['is_roaming_b']=input_df['is_roaming_b'].apply(lambda x: x if x == "ture" else "")
    input_df['network_type_l']=input_df['network_type_l'].apply(lambda x: x if x != "0" else "")
    input_df['os_version_s'] = input_df['os_version_s'].apply(lambda x: re.findall(pattern="[0-9]\\.[0-9]\\.[0-9]", string=x)[0] if len(re.findall(pattern="[0-9]\\.[0-9]\\.[0-9]", string=x)) > 0 else "")
    return input_df

def feature_transform(input_df):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    input_df['country_s'] = LabelEncoder().fit_transform(y=input_df['country_s'])


def main():
    # 读入原始数据
    p='/Users/zac/Desktop/tlv_register/part-00001-91f2e808-d9d3-4a39-952a-768f979d4ad5.csv'
    with codecs.open(p,encoding="utf-8",mode='r') as f:
        content=f.readlines()
        info = list(map(lambda x:x.strip().split("\u0394"), content))

    df = pd.DataFrame(np.array(info[1:]),columns=info[0])
    input_df = clean_data(df)
    dummy_df = pd.get_dummies(input_df)

    from sklearn.cluster import DBSCAN
    cluster_df = DBSCAN(eps=0.3, min_samples=10).fit_predict(X=dummy_df)
    # cluster_df

if __name__ == '__main__':
    pass



# 来自scala生成的数据: spark.read.parquet("/user/hive/warehouse/dw_events.db/dw_events_tlv_hour/dt=2018-06-0[1-3]/pn=*/hour=*/et=filterregister").distinct.repartition(10).write.option("header","true").option("delimiter","\u0394").mode("overwrite").csv("tlv_register")
