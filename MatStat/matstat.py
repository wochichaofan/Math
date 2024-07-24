import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Distribution:
    def __init__(self, data):
        assert type(data) in (pd.Series, np.ndarray), 'Check data type'
        self.data = self.build_series(data)
        self.dist_type = str(self).split(' ')[0][-3:]
    
    def plot_hist(self, xcol, ycol):
        plt.bar(self.data[xcol], self.data[ycol])
        plt.xticks(self.data[xcol])
        plt.show()
    
    def plot_Fx(self): 
        xs, w_ns = self.build_Fx()
        mn, mx = min([i for i in xs if i != np.NINF]), max([i for i in xs if i != np.inf])
        
        if self.dist_type == 'DRV':
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
        if self.dist_type == 'DRV':
            plt.plot(self.data.index[:-1], self.data['w'][:-1])
        else:
            plt.plot(self.data['x_i'], self.data['w_i'])
            
class DRV(Distribution):
    """
    DRV stands for Discreet random variable (Непрерывная случайная величина)
    """
    def __init__(self, data):
        super().__init__(data)
        
    def build_series(self, data):
        cols, ind = self.from_scratch(data) if len(data.shape) == 1 else self.from_dist(data)
            
        df = pd.DataFrame(cols, index=ind)
        df.loc['Total',:] = df.sum(axis=0)
        
        return df.apply(lambda x: round(x, 2))
    
    def from_scratch(self, data):
        data = pd.Series(data)
        ns = data.value_counts().sort_index()
        cols = {'n': data.value_counts().sort_index(),
               'w': ns / len(data)}
        ind = ns.index
        
        return cols, ind
    
    def from_dist(self, data):
        cols = {'n': data[-1],
               'w': data[-1] / sum(data[-1])}
        ind = data[0]
        
        return cols, ind
    
    def build_Fx(self):
        probs = self.data['w']
        idx = self.data.index[:-1]

        w_ns = [0, *list(sum(probs[:i+1]) for i in range(len(probs[:-1])))]

        iter_list = [np.NINF, *list(idx), np.inf]
        
        return iter_list, w_ns
    
class CRV(Distribution):
    """
    CRV stands for Continuous random variable (Непрерывная случайная величина)
    """
    def __init__(self, data):
        super().__init__(data)
        
    def build_series(self, data, qs=None, dist_inter='equal'):
        hs, xs, lefts, rights, ns = self.from_scratch(data, qs, dist_inter) if len(data.shape) == 1 else self.from_interval(data)

        dist_table = [[i1, i2, x, n, h] for i1, i2, x, n, h in zip(lefts, rights, xs, ns, hs)]

        df = pd.DataFrame(dist_table, columns=('left', 'right', 'x_i', 'n_i', 'h_i'))
        df['w_i'] = df['n_i'] / df['n_i'].sum()
        for col in ('n_i', 'w_i'):
            df[f'{col}/h'] = df[col] / hs
            
        return df.apply(lambda x: round(x, 2))
    
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
        probs = self.data['w_i']
        w_ns = [0, *list(sum(probs[:i+1]) for i in range(len(probs[:-1]))), 1]
        xs = list(self.data['left']) + [list(self.data['right'])[-1]]
        return xs, w_ns