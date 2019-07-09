import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pygal
from pygal.style import LightStyle

# read --- pandas
ctr = pd.read_csv("~/Desktop/ctr.csv")
ctr = ctr.sort_values('dt',ascending=False)[:15]

# pygal -- OK
bar_chart = pygal.Bar(style=LightStyle, width=800, height=600,
                      legend_at_bottom=True, human_readable=True,
                      title='ctr')

for index, row in ctr.iterrows():
    bar_chart.add(row["dt"], row["ctr3"])

bar_chart.render_to_file('ctr_pygal.svg')
bar_chart.render_to_png('ctr_pygal.png')

# read --- numpy
rawData = np.loadtxt("/Users/zac/Desktop/ctr.csv",dtype=np.str,delimiter=",")
dt = rawData[1:,-1].astype(np.str)
ctr1 = rawData[1:,5].astype(np.float)
ctr2 = rawData[1:,8].astype(np.float)
ctr3 = rawData[1:,-2].astype(np.float)

# pyplot -- OK
plt.plot(range(dt.size),ctr1,"-v",color='c')
plt.plot(range(dt.size),ctr2,"-o",color='b')
plt.plot(range(dt.size),ctr3,"-*",color='m')
# ax=plt.gca()
# for x,y in zip(range(dt.size),ctr2):
# 	ax.text(x,y,"%.4f" % y,color='r',fontsize=5)

plt.xticks(range(dt.size),dt,rotation=45)
plt.show()










