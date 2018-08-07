# encoding=utf-8
import pandas as pd
import sys
import time
import os
import json


def print_t(param):
    sys.stdout.flush()
    now = time.strftime("|%Y-%m-%d %H:%M:%S| ", time.localtime(time.time()))
    new_params = now + ": " + param
    print(new_params)
    sys.stdout.flush()


def map_v_to_idx(row, f_dict):
    for key in f_dict.keys():
        row[key] = f_dict[key][row[key]]


path_list = ["/data/houcunyue/zhoutong/data/CriteoData/test_data/train_sampled_0{idx}.txt".format(idx=i) for i in
             range(3)]
libsvm_save_directory = "/data/houcunyue/zhoutong/data/CriteoData/test_data/libsvm"
os.makedirs(libsvm_save_directory, exist_ok=True)
feature_dict_save_path = "/data/houcunyue/zhoutong/data/CriteoData/test_data/feature_dict"
delimiter = "\t"
replace_for_str_nan = "N/A"
chunk_size = 10 * 10000
numeric_cnt = 13 + 1  # 前13个特征是连续特征 + label是0/1也为numeric
col_names = ['target'] + ["feature_%s" % i for i in range(39)]

dtype_dict = {x: float for x in col_names[:numeric_cnt]}
dtype_dict.update({x: object for x in col_names[numeric_cnt:]})
na_dict = {col: 0.0 for col in col_names[:numeric_cnt]}
na_dict.update({col: replace_for_str_nan for col in col_names[numeric_cnt:]})

numeric_cols = col_names[:numeric_cnt]
ignore_cols = ['target']

feature_cnt = 0
feature_idx_dict = {}
for path in path_list:
    print_t("当前加载的原始特征文件:" + path)
    libsvm_save_path = libsvm_save_directory + "/" + os.path.split(path)[1]
    print_t("转化为libsvm格式后保存于: " + libsvm_save_path)
    _reader = pd.read_csv(path, header=None,
                          names=col_names,
                          delimiter="\t",
                          chunksize=chunk_size,
                          dtype=dtype_dict)
    i = 0
    df_list = []
    data_libsvm=[]
    for df_chunk in _reader:
        print_t("   正在处理第%s个chunk的特征" % i)
        df_chunk.fillna(na_dict, inplace=True)
        df_list.append(df_chunk)
        i += 1
        # 按列遍历,构造feature_dict
        for col in df_chunk.columns.values:
            if col in ignore_cols:
                continue
            if col in numeric_cols:
                # 此处两个 if 不能合并写为 and, 不然会导致 字典已经包含的数字型特征的列名 进入后面的 "else"
                if col not in feature_idx_dict.keys(): feature_idx_dict[col] = feature_cnt
            else:
                # 非numeric_cols
                col_unique_values = df_chunk[col].unique()
                col_feature_idx_dict = dict(
                    zip(col_unique_values, range(feature_cnt, feature_cnt + len(col_unique_values))))
                # 反过来update再赋值回去,保证同一个key使用第一次出现时的value; 目的是可以一边更新字典一边对当前的chunk编码
                if col in feature_idx_dict.keys(): col_feature_idx_dict.update(feature_idx_dict[col])
                feature_idx_dict[col] = col_feature_idx_dict
    print_t("concating ...")
    df = pd.concat(df_list,ignore_index=True)
    # 按行遍历,根据feature_dict构造libsvm数据
    print_t("构造libsvm ... ")
    for _, row in df.iterrows():
        row_libsvm = []
        for feature_name in row.index.values:
            feature_v = str(row[feature_name])
            if feature_name == "target":
                row_libsvm.append(feature_v)
            if feature_name in ignore_cols:
                continue
            if feature_name in numeric_cols:
                idx_num = str(feature_idx_dict[feature_name])
                row_libsvm.append(idx_num + ":" + feature_v)
            else:
                idx = str(feature_idx_dict[feature_name][feature_v])
                row_libsvm.append(idx + ":" + "1")
        data_libsvm.append(" ".join(row_libsvm))

    print_t("saving libsvm to: %s" % libsvm_save_path)
    with open(libsvm_save_path,"w+") as f: f.writelines(data_libsvm)


with open(feature_dict_save_path, "w+") as f: f.write(json.dumps(feature_idx_dict))
