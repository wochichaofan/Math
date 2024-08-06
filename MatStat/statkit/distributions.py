import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class distributions:
    def __init__(self, seriesClass):
        for k, v in seriesClass.__dict__.items():
            if k not in ('check_discreet', 'init_series'):
                self.__dict__[k] = copy.deepcopy(v)
        self.form_thery_table()

    def form_thery_table(self):
        df = self.table[['x', 'n']].iloc[:-1]

        total_n = self.table['n']['Total']
        h = self.table['h'][0]
        zs, f_z = map(np.array, self.Gauss())
        n_tilda = (h * total_n) / self.std * f_z
        df['z'] = zs
        df['f_z'] = f_z
        df['n\''] = n_tilda
        df = df.apply(lambda x: round(x, 4))
        self.normal_df = df
        return df

    def Gauss(self):
        z = (self.table['x'].iloc[:-1] - self.mean) / self.std
        f_z = np.exp(-(z**2)/2) / np.sqrt( 2*np.pi )
        return z, f_z

    def plot(self):
        ddf = self.normal_df
        plt.bar(ddf['x'], ddf['n'])

        plt.plot(ddf['x'], ddf['n\''], color='r')
        plt.scatter(ddf['x'], ddf['n\''], color='r')