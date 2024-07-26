class Plotter:
    def __init__(self, seriesClass):
        self = seriesClass
        
    def plot_hist(self, xcol, ycol):
        plt.bar(self.table[xcol], self.table[ycol])
        plt.xticks(self.table[xcol])
        plt.show()
    
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