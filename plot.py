import pandas as pd
from types import SimpleNamespace
import matplotlib.pyplot as plt
import h5py
import json

def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('dataset_01', df)
    store.get_storer('dataset_01').attrs.metadata = kwargs
    store.close()

def h5load(filename):
    with pd.HDFStore(filename) as store:
        for s in store.keys():
            data = store[s]
            metadata = store.get_storer(s).attrs.metadata
            print(s)
            print(data)
            print(metadata)

filename = 'data.hdf5'
h5load(filename)


def plot_minmax(df):
    """Print min max plot of DQN Agent analytics

    Params
    ======
        df :    Dataframe with scores
    """
    ax = df.plot(x='episode', y='mean')
    plt.fill_between(x='episode', y1='min', y2='max',
                     color='lightgrey', data=df)
    x_coordinates = [0, 150]
    y_coordinates = [30, 30]
    plt.plot(x_coordinates, y_coordinates, color='red')
    plt.show()


#plot_minmax(df)
