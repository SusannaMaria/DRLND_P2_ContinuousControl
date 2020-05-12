import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import pandas as pd
from types import SimpleNamespace

# define some random data that emulates your indeded code:
NCURVES = 10
np.random.seed(101)
curves = [np.random.random(20) for i in range(NCURVES)]
values = range(NCURVES)

fig = plt.figure()
ax = fig.add_subplot(111)
# replace the next line 
#jet = colors.Colormap('jet')
# with
jet = cm = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

datas =[]
filenames = ['data_ddpg.hdf5','data_td3.hdf5']
for fn in filenames:
    with pd.HDFStore(fn) as store:
        for s in store.keys():
            data = store[s]
            metadata = store.get_storer(s).attrs.metadata
            datas.append(data)
            print(s)
            print(data)
            print(metadata)



lines = []
for idx in range(len(datas)):
    line = datas[idx]
    colorVal = scalarMap.to_rgba(values[idx])
    colorText = (
        'color: (%4.2f,%4.2f,%4.2f)'%(colorVal[0],colorVal[1],colorVal[2])
        )
    retLine =  line.plot(x='episode', y='mean', color=colorVal, label=colorText)
    # retLine, = ax.plot(line,
    #                    color=colorVal,
    #                    label=colorText)
    lines.append(retLine)
#added this to get the legend to work
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right')
ax.grid()
plt.show()

