import numpy as np
from statkit.functions import *

class General_set_stats(object):
    def __init__(self, from_series=True, seriesStatsClass=None):
        if from_series:
            assert seriesStatsClass, 'If choosen to build from series, must send seriesStatsClass to input'
            self.data = seriesStatsClass.data
            self.get_stats_from_series()

    def get_stats_from_series():
        pass
    
    def base_mean_confidence(self):
        pass

    def count_N_gen(self, n, perc_n):
        pass
    
    def base_mean_interval(self, prob, mean, n, s=None, disp=None, std=None, gen_std=None):
        if not gen_std:
            trust = tinv(prob, n)
            if s:
                std = s #????
            if std:
                std = np.sqrt(n/(n-1)) * std
            if disp:
                std = np.sqrt(n/(n-1)*disp)
        else:
            trust = Laplace_table_inverse(prob)
            std = gen_std
    
        delta = (trust * std) / np.sqrt(n)
        left, right = map(stat_round, (mean-delta, mean+delta))
        return left, right, delta

    def disp_interval(self, prob, n, s=None, disp=None, std=None):
        if not s:
            if std:
                s = np.sqrt(n/(n-1))*std
            elif disp:
                s = np.sqrt(n/(n-1) * disp)
            
        alpha_left, alpha_right, k = (1-prob)/2, (1+prob)/2, n-1
        border = lambda alpha: ((n-1) * s**2) / chiinv(alpha, n)
        borders = (border(a) for a in (alpha_left, alpha_right))
        
        left, right = map(stat_round, map(np.sqrt, borders))
        
        return left, right, alpha_right-alpha_left

    def gen_mean_un_repeat(self, n, disp_2, mean, prob=.9974, N=None, perc_n=None, kind='unrepeat'):
        assert perc_n or N, 'need input data for calculating N general'
        if not N:
            N = n / perc_n
            
        if n < 30:
            trust = tinv(prob, n)
        else:
            trust = Laplace_table_inverse(prob)
        
        if kind == 'unrepeat':
            mu = np.sqrt( (disp_2/n) * (1-n/N) )
        if kind == 'repeat':
            mu = np.sqrt( disp_2/n )
            
        delta = trust * mu
        borders = mean - delta, mean + delta
        left, right = map(stat_round, (borders))
        return left, right, delta

    def rate_disp(self, disp_n, n):
        return n/(n-1) * disp_n

    def share_gen_score(self, n, k, prob=.9974, N=None, perc_n=.05, kind='unrepeat'):
        assert perc_n or N, 'need input data for calculating N general'
        if not N:
            N = n / perc_n
        
        w = k/n
        
        trust = Laplace_table_inverse(prob)
    
        if kind == 'unrepeat':
            mu = np.sqrt( ((w*(1-w))/n) * (1-n/N) )
        if kind == 'repeat':
            mu = np.sqrt( (w*(1-w))/n )
    
        delta = trust * mu
        
        borders = w - delta, w + delta
        left, right = map(stat_round, (borders))
        return left, right, delta