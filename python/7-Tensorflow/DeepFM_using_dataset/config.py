import time
import tensorflow as tf
import json


#***** 存跃的数据（midas） *****
class config_midas(object):
    # input
    _basePath = "/home/zhoutong/data/apus_ad/tfrecord_2018-09-21_to_2018-10-04_and_2018-10-05_to_2018-10-11"
    train_tfrecord_file = _basePath+"/train.tfrecord.gz"
    valid_tfrecord_file = _basePath+"/valid.tfrecord.gz"
    info_file = _basePath+"/info.json"
    # output
    base_save_dir = "/home/zhoutong/tf_modelInfo/type={type}".format(type="midas")
    # load-json
    with open(info_file,"r+") as f:
        info = "".join(f.readlines())
        result = json.loads(info)

    fieldInfo = result['allField']
    statisticInfo = result['statistic']
    tmp_map_num_f = result['numericFieldMap']#{'ad_info__budget_unit':1291744}
    max_numeric = result['numericMax']#{"ad_info__budget_unit": 2.0}

    # 连续特征的索引号要单独给出来，方便后续构造idx_sparse_tensor
    # 关于这里的filter: spark处理空数组生成JSON的问题, Seq().mkString 仍会产生一个空串，在这里要去除掉
    data_param_dicts = {
        "global_numeric_fields":list(filter(lambda x: x!="", fieldInfo['numeric_fields'].split(","))),
        "global_multi_hot_fields":list(filter(lambda x: x!="", fieldInfo['multi_hot_fields'].split(","))),
        "global_all_fields" : list(filter(lambda x: x!="", fieldInfo['all_fields'].split(","))),
        "tmp_map_num_f": result['numericFieldMap'],
        "max_numeric" : result['numericMax']

    }
    # 如果没有使用numeric 或者 multi_hot特征,会自动构造一个不起作用的numeric(multi_hot)特征,所以size要置为1
    data_param_dicts["numeric_field_size"] = len(data_param_dicts['global_numeric_fields']) if len(data_param_dicts['global_numeric_fields']) >0 else 1
    data_param_dicts["multi_hot_field_size"] = len(data_param_dicts['global_multi_hot_fields']) if len(data_param_dicts['global_multi_hot_fields']) >0 else 1


    # 调参修正如下参数
    deepfm_param_dicts = {
        "dropout_fm" : [1.0, 1.0],
        "dropout_deep" : [0.8, 0.9, 0.9, 0.9, 0.9],
        "feature_size ": statisticInfo['feature_size']+1,
        "batch_size":1024*3,
        "embedding_size": 2,
        "epoch":8,
        "deep_layers_activation" : tf.nn.relu,
        "batch_norm_decay": 0.9,
        "deep_layers":[16,8,4,2],
        "learning_rate": 0.001,
        "l2_reg":0.001
    }

    random_seed=2017
    gpu_num=1
    is_debug=False

    # @staticmethod
    # def get_now():
    #     return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    # @staticmethod
    # def get_dict(instance:object):
    #     keys = [attr for attr in dir(instance) if not callable(getattr(instance, attr)) and not attr.startswith("__")]
    #     return {key:getattr(instance,key) for key in keys}


#***** 国盛的数据（starksdk） *****
class config_starksdk(object):
    basePath = "/home/zhoutong/data/apus_ad/starksdk/tfrecord_2018-09-21_to_2018-10-04_and_2018-10-05_to_2018-10-11"
    train_tfrecord_file = basePath+"/train.tfrecord.gz"
    valid_tfrecord_file = basePath+"/valid.tfrecord.gz"
    info_file = basePath+"/info.json"
    # fields
    global_all_fields = ["label","user_profile_app__install_app_list","ad_info__advertiser","ad_info__advertiser_gp_class",
    "user_profile_basic__dpi","user_profile_basic__network_type","user_profile_basic__locale",
    "user_profile_basic__manufacturer","user_profile_basic__model","user_profile_basic__country"]
    global_numeric_fields = []
    global_multi_hot_fields = ['user_profile_app__install_app_list']
    global_one_hot_fields = []
    for i in global_all_fields:
        if i not in global_numeric_fields and i not in global_multi_hot_fields and i != "label":
            global_one_hot_fields.append(i)
    feature_size = 1087368+1
    # field_size ( 如果没有该类型的特征，后续会默认填充一个为0的tensor，这里需要把size置为1)
    multi_hot_field_size = 1 if len(global_multi_hot_fields)==0 else len(global_multi_hot_fields)
    numeric_field_size = 1 if len(global_numeric_fields)==0 else len(global_numeric_fields)
    one_hot_field_size = 1 if len(global_one_hot_fields)==0 else len(global_one_hot_fields)

    # 连续特征的索引号要单独给出来，方便后续构造idx_sparse_tensor
    tmp_map_num_f = {'ad_info__budget_unit':0}
    max_numeric = {"ad_info__budget_unit": 2.0}
    # shape
    global_dense_shape = [1024,feature_size]

    # deepfm_param
    embedding_size = 8
    dropout_fm = [1.0, 1.0]
    deep_layers = [400, 400, 400, 400]
    dropout_deep = [0.8, 0.9, 0.9, 0.9, 0.9]
    deep_layers_activation = tf.nn.relu
    epoch=10
    batch_size= 1024*6
    learning_rate= 0.001
    optimizer_type="adam"
    batch_norm= 1
    batch_norm_decay= 0.9
#     l2_reg=0.001
    verbose= True
    random_seed=2017
    gpu_num=1
    is_debug=False

    summary_save_dir = "/home/zhoutong/starksdk_summary"
    model_save_dir = "/home/zhoutong/starksdk_model"

    @staticmethod
    def get_now():
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))

    @staticmethod
    def get_dict(instance:object):
        keys = [attr for attr in dir(instance) if not callable(getattr(instance, attr)) and not attr.startswith("__")]
        return {key:getattr(instance,key) for key in keys}
