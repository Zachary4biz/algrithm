# encoding=utf-8
from hashlib import sha1
import hmac
import base64
import time , datetime
import requests
import json
import pickle


# 使用cos存储的python-sdk
def use_cos_sdk():
	from qcloud_cos import CosConfig
	from qcloud_cos import CosS3Client
	from qcloud_cos import CosServiceError
	import sys
	import logging
	logging.basicConfig(level=logging.INFO, stream=sys.stdout)

	secret_id = 'AKIDGO0YvQ1VnuoW1LxN6g4OYTFqGYhe44Wm'      # 替换为用户的 secretId
	secret_key = 'GLGpIZgcMuFOL6oQq5A7xc9W7iBSXQju'      # 替换为用户的 secretKey
	region = 'ap-shanghai'     # 替换为用户的 Region
	token = ''                  # 使用临时密钥需要传入 Token，默认为空，可不填
	config = CosConfig(Secret_id=secret_id, Secret_key=secret_key, Region=region, Token=token)
	# 2. 获取客户端对象
	client = CosS3Client(config)
	# 3. 上传文件
	response = client.upload_file(
		Bucket='zachblog-1256781535',
		LocalFilePath='/Users/zac/Desktop/9900_25.8436378601_0.1.jpg',
		Key='fireforx_mimic.jpg',
		PartSize=10,
		MAXThread=10
	)
	print(response['ETag'])
	# 4. 获取Bucket存储桶下所有文件
	try:
		response_2 = client.list_objects(
			Bucket='zachblog-1256781535',
			Delimiter='string',
			Marker='string',
			MaxKeys=100,
			Prefix='string',
			EncodingType='url'
		)
	except CosServiceError as e:
		print(e.get_origin_msg())
		print(e.get_digest_msg())
		print(e.get_status_code())
		print(e.get_error_code())
		print(e.get_error_msg())
		print(e.get_resource_location())
		print(e.get_trace_id())
		print(e.get_request_id())


# 生成授权签名
def generate_authorization(begin_time="2018-07-08 21:29:00", expire_time="2018-07-13 21:29:00" ,app_id = "1256781535",secret_id = "AKIDGO0YvQ1VnuoW1LxN6g4OYTFqGYhe44Wm", secret_key = "GLGpIZgcMuFOL6oQq5A7xc9W7iBSXQju"):
	# #### 官方示例
	# original_official = "a=1252821871&b=tencentyun&k=AKIDgaoOYh2kOmJfWVdH4lpfxScG2zPLPGoK&e=1438669115&t=1436077115&r=11162&u=0&f="
	# # 注意不能用 hexdigest()
	# sign_temp_official = hmac.new("nwOKDouy5JctNOlnere4gkVoOUz5EYAb",original_official,sha1).digest()
	# # 加密结果: \xa7f9\x88\x862\x06d\r}K\xcf{w\xb5\xb3\x11\r\xfe\xb6
	# sign_official = base64.b64encode(sign_temp_official+original_official)
	# # 最终结果: p2Y5iIYyBmQNfUvPe3e1sxEN/rZhPTEyNTI4MjE4NzEmYj10ZW5jZW50eXVuJms9QUtJRGdhb09ZaDJrT21KZldWZEg0bHBmeFNjRzJ6UExQR29LJmU9MTQzODY2OTExNSZ0PTE0MzYwNzcxMTUmcj0xMTE2MiZ1PTAmZj0=

	# 开始时间时间 秒为单位
	time_now = time.mktime(time.strptime(begin_time, '%Y-%m-%d %H:%M:%S'))
	# 过期时间 秒为单位
	time_expire = time.mktime(time.strptime(expire_time, '%Y-%m-%d %H:%M:%S'))
	e = str(long(time_expire))
	t = str(long(time_now))
	r = "19950121" # random_seed
	u="0"
	f=""
	original = "a=%s&k=%s&e=%s&t=%s&r=%s&u=%s&f=%s" % (app_id,secret_id,e,t,r,u,f)
	sign_temp = hmac.new(secret_key,original,sha1).digest()
	sign = base64.b64encode(sign_temp+original)
	return sign

sign = generate_authorization("2018-06-29 21:")
img_url = "https://s1.ax2x.com/2018/06/04/RBZ8N.png"
img_url2="https://zachblog-1256781535.cos.ap-shanghai.myqcloud.com/OCR%E6%B5%8B%E8%AF%95%E5%9B%BE%E7%89%87.png"

def send_to_ocr(img_url_input,sign_input):
	url = "https://recognition.image.myqcloud.com/ocr/general"
	header={
	'Authorization':sign_input,
	'Host': 'recognition.image.myqcloud.com',
	'content-type' : 'application/json'
	}
	body = {
	'appid' : '1256781535',
	'url' : img_url_input
	}
	result = requests.post(url,headers=header,data=json.dumps(body))
	result_dict = result.json() # result_dict.keys : [u'message', u'code', u'data']
	with open("ocr_result_dict.pk","wb") as f:
		pickle.dump(result_dict,f)
	if result.status_code == 200:
		# result_data = result_dict['data'] # result_data : [u'items', u'angle', u'class', u'session_id']
		# result_list = result_data['items']
		result_list = result.json()['data']['items']
		for line in map(lambda x: x['words'], result_list):
			dict_form = dict()
			for x in line:
				dict_form[x['confidence']]=x['character']
			print(min(dict_form.keys()),dict_form[min(dict_form.keys())])
		ocr_result = "\n".join(map(lambda x: x['itemstring'],result_list))
		with open("ocr_result.txt",'w+') as f:
			for i in ocr_result:
				f.write(i)
		print(ocr_result)
		return ocr_result
	else:
		return result















