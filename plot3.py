
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