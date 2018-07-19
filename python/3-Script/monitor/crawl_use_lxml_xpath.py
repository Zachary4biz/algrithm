# encoding=utf-8
import requests
from lxml import html


def get_category(pn,output):
	url = "https://play.google.com/store/apps/details?id=%s" % pn
	response=requests.get(url)
	selector = html.fromstring(response.content)
	result = selector.xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "R8zArc", " " ))]')
	category = map(lambda x: x.get('href').split("/")[-1],result)
	output.append((pn,category))

output=[]
to_search_pn = ["com.skype.raider","com.google.android.apps.maps","com.snapchat.android"]
for pn in to_search_pn:
	get_category(pn,output)























