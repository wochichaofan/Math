import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator
from math import factorial as fact
from statkit.functions import *

class distributions:
    def __init__(self, seriesClass):
        for k, v in seriesClass.__dict__.items():
            if k not in ('check_discreet', 'init_series'):
                self.__dict__[k] = copy.deepcopy(v)

    def fix_df(self, df):
        is_to_sum = [i for i in df.index if df.iloc[i]['n']<5]
        merged_row = reduce(operator.__add__, (df.iloc[i] for i in is_to_sum))
        filtered_df = df.iloc[[i for i in df.index if i not in is_to_sum]]
        
        new_df = pd.concat([filtered_df, pd.DataFrame(merged_row).T])
        return new_df 
        
    def Pearson_chisq_test(self, alpha, r):
        data = self.fix_df(self.normal_df)
        m = len(data)
        k = m - r - 1
        chi2_crit = chi2inv(alpha, k, prec=False)
        chi2_observ = sum(((data['n'] - data['n\''])**2)/data['n\''])
        res = 'did' if chi2_observ <= chi2_crit else 'didnt'
        print(f'Chi2_crit = {chi2_crit}; Chi2_observed = {chi2_observ}')
        print(f'At the level of importance of {alpha} H_o hypothesis about normal distribution of the series {res} prove itself.')
    
    def plot(self):
        ddf = self.normal_df
        plt.bar(ddf['x'], ddf['n'])

        plt.plot(ddf['x'], ddf['n\''], color='r')
        plt.scatter(ddf['x'], ddf['n\''], color='r')

class poissonDistribution(distributions):
    def __init__(self, seriesClass):
        super().__init__(seriesClass)
        self.form_theory_table()

    def conduct_test(self, alpha):
        self.Pearson_chisq_test(alpha, r=1)
    
    def form_theory_table(self):
        df = self.table[['x', 'n']].iloc[:-1]
        xs = df['x']
        ns = df['n']
        total_n = self.table['n']['Total']
        
        p_i = (self.mean**xs / np.array([fact(int(x)) for x in xs])) * np.exp(-self.mean)
        n_tilda = p_i * total_n
        df['p(i)'] = p_i
        df['n\''] = n_tilda
        df = df.apply(lambda x: round(x, 4))
        self.normal_df = df
        return df

class normalDistribution(distributions):
    def __init__(self, seriesClass):
        super().__init__(seriesClass)
        self.form_theory_table()

    def conduct_test(self, alpha):
        self.Pearson_chisq_test(alpha, r=2)
    
    def form_theory_table(self):
        df = self.table[['x', 'n']].iloc[:-1]

        total_n = self.table['n']['Total']
        if 'h' not in self.table.columns:
            h = abs(sum(self.table['x'][:-1])/reduce(operator.__sub__, self.table['x'][:-1]))
        else:
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
