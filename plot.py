import pandas as pd
from types import SimpleNamespace
import matplotlib.pyplot as plt
import h5py
import json
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]


print(sorted_names)

def plot_minmax(dfs):
    """Print min max plot of DQN Agent analytics

    Params
    ======
        df :    Dataframe with scores
    """
    coln=15
    for df in dfs:
        if coln==15:
            # , color=scalarMap.to_rgba(coln)
            ax = df.plot(x='episode', y='mean', label='ddpg', color='red', alpha=0.7 )
            df.plot(ax=ax, x='episode', y='std', label='std ddpg', color='grey', alpha=0.7)
            plt.fill_between(x='episode', y1='min', y2='max',
                             data=df, color='indianred', alpha=0.7)
        else:
            df.plot(ax=ax, x='episode', y='mean', label='td3', color='green' , alpha=0.7)
            df.plot(ax=ax, x='episode', y='std', label='std td3', color='steelblue' , alpha=0.7)
            plt.fill_between(x='episode', y1='min', y2='max',
                             data=df, color='lightgreen', alpha=0.7)
    
        coln = coln + 2

    x_coordinates = [0, 150]
    y_coordinates = [30, 30]
    plt.plot(x_coordinates, y_coordinates, color='red',linestyle='--')

def plot_minmax2(df):
    """Print min max plot of DQN Agent analytics

    Params
    ======
        df :    Dataframe with scores
    """
    # , color=scalarMap.to_rgba(coln)
    ax = df.plot(x='episode', y='mean', label='ddpg', color='red', alpha=0.7 )
    df.plot(ax=ax, x='episode', y='std', label='std ddpg', color='grey', alpha=0.7)
    plt.fill_between(x='episode', y1='min', y2='max',
                        data=df, color='indianred', alpha=0.7)

    x_coordinates = [0, 150]
    y_coordinates = [30, 30]
    plt.plot(x_coordinates, y_coordinates, color='red',linestyle='--')





def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('dataset_01', df)
    store.get_storer('dataset_01').attrs.metadata = kwargs
    store.close()

def h5load(filenames):
    datas =[]
    for fn in filenames:
        with pd.HDFStore(fn) as store:
            for s in store.keys():
                data = store[s]
                metadata = store.get_storer(s).attrs.metadata
                datas.append(data)
                print(s)
                print(data)
                print(metadata)
    plot_minmax(datas)

    plt.show()

def h5load2(filenames):
    datas =[]
    df = pd.DataFrame(columns=['episode', 'duration',
                               'min', 'max', 'std', 'mean'])
    for fn in filenames:
        with pd.HDFStore(fn) as store:
            i_episode = 0
            for s in store.keys():
                data = store[s]
                metadata = store.get_storer(s).attrs.metadata
                mean = data["mean"].mean()
                minv = data["min"].mean()
                maxv = data["max"].mean()
                std = data["std"].mean()
                dur = data["duration"].mean()

                episode = int(s.replace("/dataset_", ""))
                df.loc[i_episode] = [episode] + list([dur, minv, maxv, std, mean])
                i_episode+=1     
                print(int(episode))
                #print(data)
                #print(metadata)
    plot_minmax2(df)

    plt.show()


# filenames = ['data_ddpg.hdf5','data_td3.hdf5']
# h5load(filenames)

filenames = ['test_td3.hdf5']
h5load2(filenames)
