from vis import *


class MultiIndexFacilitate(object):
    def __init__(self, columns):
        levels = list(columns.levels)
        names = list(columns.names)
        name2level = {name: level for name, level in zip(names, levels)}
        name2ind = {name: ind for ind, name in enumerate(names)}
        self.levels = levels
        self.names = names
        self.labels = columns.labels
        self.names2levels = name2level
        self.names2ind = name2ind
        self.index = columns

    def update(self):
        self.index = pd.MultiIndex.from_product(self.levels,names=self.names)


def get_shape(arr):
    return np.array(arr).shape


def df2arr(df):
    df = df.copy()
    df = df.unstack()
    shape = map(len, df.index.levels)
    arr = np.full(shape, np.nan)
    arr[df.index.labels] = df.values.flat
    return arr, MultiIndexFacilitate(df.index)


def arr2df(arr, indexf):
    df2 = pd.DataFrame(arr.flatten(), index=indexf.index)
    df2.reset_index().pivot_table(values=0, index=indexf.names[-1:], columns=indexf.names[:-1])
    return df2


if __name__ == '__main__':
    visualizer = Visualizer(join='outer', stat_only=True, paranet_folder='stat401_10')
    df = visualizer.perf_df
    arr, indexf = df2arr(df)
    print df.shape, arr.shape
    df2 = arr2df(arr, indexf)
