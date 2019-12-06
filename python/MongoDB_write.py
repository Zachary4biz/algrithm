# author: zac
# create-time: 2019-11-18 10:13
# usage: - 
import pymongo
import time
import bson
import random
from tqdm.auto import tqdm
pattern = {
    "inner_type": "artifact",           # 数据类型 artifact:作品，material：素材
    "photo_id": 123456,                 # 【素材id】  【作品id】
    "_id": 1551839776248225,            # 推荐系统内部 标识
    "resource_id": 1551839776248225,    # 推荐系统内部 标识
     "is_recommend":0,                  # 是否是运营加精内容 0：非 1：是
    "is_we_media":1,                  # 是否是自媒体内容 0：非 1：是
    "is_thirdparty":1,                  # 是否是第三方素材 0：非 1：是
    "related_material_id": [],   # 作品关联的素材推荐ID
    "origin_url": "https://s3.ap-southeast-1.amazonaws.com/images.deccanchronicle.com/dc-Cover-m5eb04rfkvffshc2vhqdf5e9e2-20190305172114.jpeg",  # 图片原图url
    "banner_url": "https://s3.ap-southeast-1.amazonaws.com/images.deccanchronicle.com/dc-Cover-m5eb04rfkvffshc2vhqdf5e9e2-20190305172114.jpeg",  # 图片预览url
    "title": "",    # 标题
    "desc" :  "",   # 描述
    "country": "IN",                    # 产生国家
    "lang": "en",                       # 语言
    "resource_type": 70001,            #推荐系统内部 资源标识
    "usable_country": ["IN","USA"],     # 下发国家列表
    "pub_products": [123, 456],         # 下发产品pid列表
    "category": 3,                      # 一级分类
    "sub_category": -1,                 # 二级分类
    "subject_list": [],          # 三级分类 专题id列表
    "create_time": 1551944561,       # 接收数据时间 单位秒
    "pub_time": 1551944561,          # 发布时间 （对应图片那边的create_time）单位秒
    "data_source": 0,                   # 来源  1:客户端 2:运营创建
    "product_id": 1,                    # 所属产品pid （当data_source为 0:运营创建时 product_id = 0）
    "material_type": 1,                 # 素材类型（滤镜，抠图）
    "account": {},
    "user_order": 1,                    # 用户排序，数值1000以内，越大优先级越高
    "activity":[],
   "tags": [],
   "thirdparty_tags": [],
    "extended_info":{},
    "algo_profile": {},
    "trial": {},
    "review": {},
    "quality_grade": [],                # 图片质量   从审核拿出的字段 1-High, 2-Medium, 3-Low
    "review_status": 1,                 # 审核状态 0通过, 1不通过, 2 算法通过, 3 待审核
    "review_type": [],    # 根据审核结果来进行数据
    "status": 1,                        # 上线标识；  1：下线，0：上线
    "buss_type" : 0 # 0离线，1在线
}

# client = pymongo.MongoClient(host="10.10.88.13",port=27017)
# mongo_client=pymongo.MongoClient("mongodb://dev-mongo.apuscn.com:27017/article_repo")
# client0=pymongo.MongoClient("mongodb://10.10.88.13:27017")
# db=client['article_repo']
# 注意，这里不佳article_repo不行，会报错 Authentication failed.
client=pymongo.MongoClient("mongodb://article_repo_rw:Z5ROAsDCUwKxUFcoGtwv@content-mongodb001.apusdb.com:27060,content-mongodb002.apusdb.com:27060/article_repo")
db=client['article_repo']
# db.collection_names()
co = db['picture_info_doc']
co.find_one({'id':123})
db.article_repo.picture_info_doc.count_documents({})

batch_id = 20191204
base_id = 19930101
base_photoID = 19930101
with open("/Users/zac/Downloads/enthnicity_img_copied/resInfo.txt","r") as f:
    urls = [i.strip() for i in f.readlines()]
random.shuffle(urls)
# total = 20
# step = 5
total = len(urls)
step = 100
for i in tqdm(range(0, total, step), desc="write to Mongo"):
    t = int(time.time())
    urls_part = urls[i:i + step]
    documents = []
    for j, url in enumerate(urls_part):
        pattern.update({"_id": str(base_id + i + j),
                        "resource_id": bson.int64.Int64(base_id + i + j),
                        "photo_id": bson.int64.Int64(base_photoID + i + j),
                        "create_time": bson.int64.Int64(t),
                        "pub_time": bson.int64.Int64(t),
                        "origin_url": url,
                        "banner_url": url,
                        "batch":batch_id})
        documents.append(pattern.copy())
    # print(f"写入数据: {documents[0]['id']} 至 {documents[-1]['id']}")
    co.insert_many(documents)
client.close()
# db.picture_info_doc.find({"_id":{'$gte':'1564551600303158'}},{"photo_id":1,"banner_url":1,"buss_type":1,"batch":1,"resource_id":1,"create_time":1}).limit(1)
# db.picture_info_doc.find({"_id":{'$gte':'19920101'}},{"photo_id":1,"banner_url":1,"buss_type":1,"batch":1,"resource_id":1,"create_time":1}).limit(1)
# db.picture_info_doc.remove({"photo_id":{'$in': [19920101, 19920102,19920103,19920104,19920105,19920106,19920107,19920108,19920109,19920110,19920111,19920112,19920113,19920114,19920115,19920116,19920117,19920118,19920119,19920120]}})

