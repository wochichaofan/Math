import numpy as np
import pandas as pd

class GeneralSeries:
    def __init__(self, data):
        # assert type(data) in (pd.Series, np.ndarray), 'Check data type'
        data = np.array(data)
        self.check_discreet = lambda: 'discreet' in str(self)
        self.init_series = data
        self.table = self.build_series(data)
        # self.stats()
    
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
            
class discreetVariationSeries(GeneralSeries):
    def __init__(self, data):
        super().__init__(data)
    
    def build_series(self, data):
        cols = self.get_build_func()(data)
        df = pd.DataFrame(cols).reset_index(drop=True) #, index=ind)
        df.loc['Total',:] = df.sum(axis=0)
        
        return df.apply(lambda x: round(x, 2))
    
    def from_scratch(self, data):
        """
        Form a series from 1-dim data
        """
        data = pd.Series(data)
        ns = data.value_counts().sort_index()
        ws = ns / len(data)
        wcum = np.cumsum(ws)
        cols = {'x' : ns.index,
                'n': ns,
                'w': ws,
                'w_cum': wcum}
        # ind = ns.index
        
        return cols #, ind
    
    def from_dist(self, data):
        """
        Form a series from 2-dim data of same length
        Data should be of a kind [[x_i], [n_i]]
        """
        ns = data[-1]
        ws = data[-1] / sum(data[-1])
        wcum = np.cumsum(ws)
        cols = {'x': data[0],
                'n': ns,
                'w': ws,
                'w_cum' : wcum}
        # ind = data[0]
        
        return cols #, ind
    
    def build_Fx(self):
        probs = self.table['w']
        idx = self.table.index[:-1]

        w_ns = [0, *list(sum(probs[:i+1]) for i in range(len(probs[:-1])))]

        iter_list = [np.NINF, *list(idx), np.inf]
        
        return iter_list, w_ns
    
class intervalVariationSeries(GeneralSeries):
    def __init__(self, data):
        super().__init__(data)
        
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

    def check_interval(self, data):
        if None in data:
            mean_interval = np.mean([(i[-1] - i[0]) for i in data[1:-1][:, 0:2]])

            if None in data[0]:
                data[0][0] = int(data[0][1] - mean_interval)

            if None in data[-1]:
                data[-1][1] = int(data[-1][0] + mean_interval)

        assert None not in data, 'Missing values in data that can\'t be handled' 
        return data
    
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