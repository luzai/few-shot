from loader import Loader
from log import logger
from opts import Config
import time, numpy as np
import pandas as pd, re
import time, glob, os, os.path as osp
from datasets import Dataset
from models import VGG
from opts import Config
from log import logger


class Visualizer(object):
    def __init__(self, config_dict):
        self.aggregate(config_dict)
        self.split()

    def split(self):
        self.perf_df = self.select(self.df, 'name', "(?:val_loss|loss|val_acc|acc)")
        self.stat_df = self.select(self.df, 'name', "^obs.*")

    def select(self, df, level_name, par_name):
        sel_name = df.columns.get_level_values('name')
        f_sel_name = set()
        pat = re.compile(par_name)
        for sel_name_ in sel_name:
            judge = bool(pat.match(sel_name_))
            if judge: f_sel_name.add(sel_name_)
        df_l = []
        for sel_name_ in f_sel_name:
            df_l.append(df.xs(sel_name_, level=level_name, axis=1, drop_level=False))
        df = pd.concat(df_l, axis=1)
        df.sort_index(level=self.level_name, axis=1, inplace=True)
        return df

    def aggregate(self, conf_dict_in):
        conf_name_dict = {}
        loaders = {}
        for model_type in conf_dict_in.get('model_type', [None]):
            for lr in conf_dict_in.get('lr', [None]):
                conf = Config(epochs=231, batch_size=256, verbose=2,
                              model_type=model_type,
                              dataset_type='cifar10',
                              debug=False, others={'lr': lr}, #, 'limit_val': True
                              clean=False)
                path = Config.root_path + '/epoch/' + conf.name
                if osp.exists(path):
                    conf_name_dict[conf.name] = conf.to_dict()
                    loader = Loader(path=path)
                    loader.load()
                    loaders[conf.name] = loader
        df_l = []
        index_l = []
        assert  len(conf_name_dict)!=0,'should not be empty'
        for ind in range(len(conf_name_dict)):
            conf_name = conf_name_dict.keys()[ind]
            conf_dict = conf_name_dict[conf_name]

            loader = loaders[conf_name]
            loader.load()

            scalar = loader.scalars
            act = loader.act
            param = loader.params

            df_l.append(scalar)
            for name in scalar.columns:
                index_l.append(conf_dict.values() + [name])

            df_l.append(act)
            for name in act.columns:
                index_l.append(conf_dict.values() + [name])

            df_l.append(param)
            for name in param.columns:
                index_l.append(conf_dict.values() + [name])

        index_l = np.array(index_l).astype(basestring).transpose()
        index_name = conf_dict.keys() + ['name']
        index = pd.MultiIndex.from_arrays(index_l, names=index_name)
        df = pd.concat(df_l, axis=1, join='inner')
        df.columns = index
        df = df.sort_index(axis=1, level=index_name)
        self.level_name = index_name
        self.column_index = index
        self.df = df


if __name__ == '__main__':
    config_dict = {'model_type': ['vgg5', 'vgg11', 'vgg19'],
                   'lr': [1, 1e-2, 1e-5]}
    visualizer = Visualizer(config_dict)
    print visualizer.perf_df.columns
    print visualizer.stat_df.columns

