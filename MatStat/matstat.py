import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil

class Distribution:
    def __init__(self, data):
        assert type(data) in (pd.Series, np.ndarray), 'Check data type'
        self.check_discreet = lambda: 'discreet' in str(self)
        self.init_series = data
        self.table = self.build_series(data)
        self.stats()

    def plot_hist(self, xcol, ycol):
        plt.bar(self.table[xcol], self.table[ycol])
        plt.xticks(self.table[xcol])
        plt.show()
    
#     Make a function plot all
    
    def get_build_func(self):
        if self.check_shape():
            return self.from_scratch
        else:
            if self.check_discreet():
                return self.from_dist
            else:
                return self.from_interval    
    
    def check_shape(self):
        return len(self.init_series.shape) == 1
    
    def stats(self):
        df = self.table.drop('Total')
        total_n = self.table['n']['Total']
        
        def count_from_modal_interval():
            nonlocal df, total_n
            
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
            nonlocal df, total_n
            df['ncum'] = np.cumsum(df['n'])

            med_base = ceil(total_n / 2)
            med_index = list(filter(lambda x: df['ncum'][x] > med_base, df.index[:-1]))[0]

            left_median_border = df.iloc[med_index]['left']
            left_interval_ncum = 0 if med_index == 0 else df['ncum'][med_index-1]
            h = df.iloc[med_index]['right'] - left_median_border
            n = df.iloc[med_index]['n']

            median = left_median_border + (.5 * total_n - left_interval_ncum) / n * h
            return median
        
        
        if self.check_discreet():
            self.mean = sum(df.index * df['n']) / total_n
            self.mode = max(df.index, key=lambda x: df['n'][x])
            self.median = min(df[df['w_cum'] >= .5].index)
        else:
            self.mean = sum(df['x'] * df['n']) / total_n
            self.mode = count_from_modal_interval()
            self.median = count_from_median_interval()
    
    def show_stats(self):
        names = ('mean', 'mode', 'median')
        vals = map(lambda x: round(x, 4), (self.mean, self.mode, self.median))
        dict_ = {n: [v] for n, v in zip(names, vals)}
        
        return pd.DataFrame.from_dict(dict_)
    
    def plot_Fx(self): 
        xs, w_ns = self.build_Fx()
        mn, mx = min([i for i in xs if i != np.NINF]), max([i for i in xs if i != np.inf])
        
        if self.check_discreet():
            for prob, i, j in zip(w_ns, xs[:-1], xs[1:]):
                x1 = mn-10e5 if i == np.NINF else i
                x2 = mx+10e5 if j == np.inf else j
                print('{:^4}, {:^5} < x <= {:^5}'.format(round(prob, 4), i, j))
                plt.plot([x1, x2], [prob]*2, label=fr'x $ \in $ ({i}; {j})', linewidth=2)
            plt.legend()
                
        else:
            plt.plot([mn-10e5, mn], [0]*2, color='orange', linewidth=2)
            plt.plot(xs, w_ns, color='orange', linewidth=2)
            plt.scatter(xs, w_ns)
            plt.plot([mx, mx+10e5], [1]*2, color='orange', linewidth=2)

        plt.xlim([mn-1, mx+1])
        plt.grid(True)
        plt.axhline(0, color='k', linewidth=.5)
        plt.axvline(0, color='k', linewidth =.5)
        
    def plot_polygon(self):
        if self.check_discreet():
            plt.plot(self.table.index[:-1], self.table['w'][:-1])
        else:
            plt.plot(self.table['x'][:-1], self.table['w'][:-1])
            
            
class discreetVariationSeries(Distribution):
    def __init__(self, data):
        super().__init__(data)
        # self.stats()
    
    def build_series(self, data):
        cols, ind = self.get_build_func()(data)
        df = pd.DataFrame(cols, index=ind)
        df.loc['Total',:] = df.sum(axis=0)
        
        return df.apply(lambda x: round(x, 2))
    
    def from_scratch(self, data):
        data = pd.Series(data)
        ns = data.value_counts().sort_index()
        ws = ns / len(data)
        wcum = np.cumsum(ws)
        cols = {'n': ns,
               'w': ws,
               'w_cum': wcum}
        ind = ns.index
        
        return cols, ind
    
    def from_dist(self, data):
        ns = data[-1]
        ws = data[-1] / sum(data[-1])
        wcum = np.cumsum(ws)
        cols = {'n': ns,
               'w': ws,
               'w_cum' : wcum}
        ind = data[0]
        
        return cols, ind
    
    def build_Fx(self):
        probs = self.table['w']
        idx = self.table.index[:-1]

        w_ns = [0, *list(sum(probs[:i+1]) for i in range(len(probs[:-1])))]

        iter_list = [np.NINF, *list(idx), np.inf]
        
        return iter_list, w_ns
    
class intervalVariationSeries(Distribution):
    def __init__(self, data):
        super().__init__(data)
        
    def check_interval(self, data):
        mean_interval = np.mean([(i[-1] - i[0]) for i in data[1:-1][:, 0:2]])

        if None in data[0]:
            data[0][0] = int(data[0][1] - mean_interval)

        if None in data[-1]:
            data[-1][1] = int(data[-1][0] + mean_interval)

        assert None not in data, 'Missing values in data that can\'t be handled' 
        return data
        
    def build_series(self, data, qs=None, dist_inter='equal'):
        data = self.check_interval(data)
        hs, xs, lefts, rights, ns = self.from_scratch(data, qs, dist_inter) if len(data.shape) == 1 else self.from_interval(data)

        dist_table = [[i1, i2, x, n, h] for i1, i2, x, n, h in zip(lefts, rights, xs, ns, hs)]

        df = pd.DataFrame(dist_table, columns=('left', 'right', 'x', 'n', 'h'))
        df['w'] = df['n'] / df['n'].sum()
        for col in ('n', 'w'):
            df[f'{col}/h'] = df[col] / hs
        df['w_cum'] = np.cumsum(df['w'])
        # df['ncum'] = np.cumsum(df['n'])
            
        df.loc['Total',:] = df.sum(axis=0)
            
        return df
    # .apply(lambda x: round(x, 2))
    
    def from_scratch(self, data, qs, dist_inter):
        sturges = lambda data: int(1 + 3.322 * np.log10(len(data)))
        count_xi = lambda i1, i2: round((i1+i2)/2, 2)
        
        if not qs:
            qs = sturges(data)
        
        if dist_inter == 'equal':
            h = (max(data) - min(data)) / qs
            limits = [min(data)] + [min(data) + _*h for _ in range(1, qs+1)]
            h = [h] * qs
        else:
            h = list() # complete for unequal distributions

        ns, xs, lefts, rights = [list() for _ in range(4)]
        for i1, i2 in zip(limits[:-1], limits[1:]):
            lefts.append(i1)
            rights.append(i2)
            ns.append(len(list(filter(lambda x: i1 <= x < i2 + .0001, data))))
            xs.append(count_xi(i1, i2))
        
        return h, xs, lefts, rights, ns
        
    def from_interval(self, data):
        """
        Input data should have a following structure: [[left, right, n_i], ...]
        """
        lefts = data[:, :1]
        rights = data[:, 1:-1]
        ns = data[:, -1]
        xs = (lefts + rights) / 2
        h = rights - lefts
        return map(np.ndarray.flatten, (h, xs, lefts, rights, ns))
    
    def build_Fx(self):
        probs = self.table['w'][:-1]
        w_ns = [0, *list(sum(probs[:i+1]) for i in range(len(probs[:-1]))), 1]
        xs = list(self.table['left'][:-1]) + [list(self.table['right'][:-1])[-1]]
        return xs, w_ns