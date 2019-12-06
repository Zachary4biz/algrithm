import requests
import datetime
import os
import json
import time

url = "http://fex-api.apuscn.com/internal/api/object"
dt = datetime.datetime.now().strftime("%Y-%m-%d")


base_dir = "/Users/zac/Downloads/enthnicity_img_copied/train"
img_file = [os.path.join(root,name) for root, dirs, files in os.walk(base_dir) for name in files if not name.startswith(".")]

res_list = []
with open("/Users/zac/Downloads/resInfo.txt","w+") as fw:
    for i in img_file[:5]:
        f_path = f"label_face_img/{dt}/{os.path.split(i)[1]}"
        res = requests.post(url+"?token="+"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwbGF0Zm9ybSI6ImJpZ2RhdGEtaW1nIiwiaWF0IjoxNTc0MDQ1ODIxfQ.kMnrA94EkDYpsbhIwIRHX-Y_6gEz88ug6aVj8JA1rhE",
                            data={
                                'prefix': "bigdata-img",
                                'filePath': f_path},
                            files={"file":open(i,"rb")})
        if res.status_code == 200:
            result_url="https://thumbor.subcdn.com/imageView2/" + json.loads(res.text)['data']['url']
        else:
            print(f"img upload fail at : {i}.")
        time.sleep(0.2)
        fw.writelines(result_url+"\n")
