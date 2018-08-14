# encoding=utf-8
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# 解决plt中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决黑体不显示负号的问题

import pandas as pd
pd.set_option('display.height',1000)
pd.set_option('display.width',800)
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',12)

path = "/Users/zac/5-Algrithm/python/8-DataVisualize/sample/info.csv"
df = pd.read_csv(path,delimiter="\t")


#
# for i in all_descriptions[:20]:
# 	target_df = data_ori[data_ori['rule'] == i].sort_values(by=['dt'])
# 	target_df.plot(figsize=(14,6),
# 				   sharex=True,
# 				   fontsize=7,
# 				   grid=True,
# 				   subplots=True,
# 				   layout=(3,3),
# 				   title=i,
# 				   x='dt',y=numeric_columns)
# 	plt.show()
