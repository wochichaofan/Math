import pandas as pd
import numpy as np
from math import ceil
from statkit.functions import *

class Stats(object):
    def __init__(self, seriesClass):
        self.check_discreet = 'discreet' in str(seriesClass)
        self.data = seriesClass.table
        self.count_abs_stats()
        self.count_rel_stats()
        self.form_stats_df()
    
    def count_abs_stats(self):
        df = self.data.drop('Total')
        total_n = self.data['n']['Total']
        x, n = df['x'], df['n']
        
        if self.check_discreet:
            self.discreet_stats(df, total_n)
        else:
            self.interval_stats(df, total_n)
        self.range = max(x) - min(x)
        self.abs_dev = (1/total_n) * sum(abs(x - self.mean) * n)
        self.disp = sum(x**2*n)/total_n - (sum(x*n)/total_n)**2 #(1/total_n) * sum((df['x'] - self.mean)**2 * df['n'])
        self.UB_disp = total_n/(total_n-1) * self.disp
        self.std = np.sqrt(self.disp)
        self.UB_std = np.sqrt(self.UB_disp)
    
    def interval_stats(self, df, total_n):
        def count_from_modal_interval():
            # nonlocal df, total_n
            
            indxs = df.index[:-1]
            modal_ind = max(df.index[:-1], key=lambda x: df['n'][x])
            modal = df.iloc[modal_ind]['n']
            left_modal = 0 if modal_ind-1 < 0 else df.iloc[modal_ind-1]['n']
            right_modal = 0 if modal_ind+1 > max(indxs) else df.iloc[modal_ind+1]['n']
            x_0  = df.iloc[modal_ind]['left']
            h = df.iloc[modal_ind]['right'] - x_0

            mode = x_0 + (modal - left_modal)/((modal - left_modal) + (modal - right_modal)) * h

            return mode
        
        def count_from_median_interval():
            # nonlocal df, total_n
            df['ncum'] = np.cumsum(df['n'])

            med_base = ceil(total_n / 2)
            med_index = list(filter(lambda x: df['ncum'][x] > med_base, df.index[:-1]))[0]

            left_median_border = df.iloc[med_index]['left']
            left_interval_ncum = 0 if med_index == 0 else df['ncum'][med_index-1]
            h = df.iloc[med_index]['right'] - left_median_border
            n = df.iloc[med_index]['n']

            median = left_median_border + (.5 * total_n - left_interval_ncum) / n * h
            return median
        
        self.mean = sum(df['x'] * df['n']) / total_n
        self.mode = count_from_modal_interval()
        self.median = count_from_median_interval()
        
        
    def discreet_stats(self, df, total_n):
        self.mean = sum(df['x'] * df['n']) / total_n
        self.mode = df.iloc[max(df.index, key=lambda x: df['n'][x])]['x']
        self.median = df['x'][min(df[df['w_cum'] >= .5].index)]
    
    def count_rel_stats(self):
        self.СV = self.std/self.mean * 100

    def form_stats_df(self):
        names = ('mean', 'mode', 'median', 'range', 'abs_dev', 'disp', 'UB disp', 'std', 'UB_std', 'coeff var')
        vals = (self.mean, self.mode, self.median, self.range, self.abs_dev, self.disp, self.UB_disp, self.std, self.UB_std, self.СV)
        round_vals = map(lambda x: round(x, 2), vals)
        dict_ = {n: [v] for n, v in zip(names, round_vals)}
        self.stats_data = pd.DataFrame.from_dict(dict_)
    
    def show_stats(self):
        return self.stats_data