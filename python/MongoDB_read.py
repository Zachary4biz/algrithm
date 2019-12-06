# author: zac
# create-time: 2019-11-28 15:32
# usage: - 
import pymongo
import time
import itertools
from tqdm.auto import tqdm
from zac_pyutils import ExqUtils
f_iter = ExqUtils.load_file_as_iter("/Users/zac/Downloads/info.csv")
client=pymongo.MongoClient("mongodb://article_repo_rw:Z5ROAsDCUwKxUFcoGtwv@content-mongodb001.apusdb.com:27060,content-mongodb002.apusdb.com:27060/article_repo")
co = client['article_repo']['picture_info_doc']
print("find_one",co.find_one())
with open("/Users/zac/Downloads/info_with_url.csv","w+") as fw:
    for i in tqdm(f_iter):
        i = i.strip()
        picId=i.split(",")[0]
        try:
            url = co.find_one({'_id': picId},{'banner_url':1})['banner_url']
            fw.writelines(f"{i},{url}\n")
        except Exception as e:
            fw.writelines(f"{i},{None}\n")
client.close()
