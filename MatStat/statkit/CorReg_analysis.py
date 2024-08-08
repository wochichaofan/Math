from statkit import seriesBuilder as SB
from statkit.distributions import *

class CRA:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)

    def plot_scatter(self):
        Xrange = max(self.X) - min(self.X) 
        Yrange = max(self.Y) - min(self.Y)
        
        fig, ax = plt.subplots()
        
        plot = ax.scatter(self.X,self.Y)
        
        ax.yaxis.grid(True, color ="lightgray")
        ax.xaxis.grid(True, color ="lightgray")
        # ax.set_axisbelow(True)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        ax.set_title('Title', fontsize = 12, fontweight ='bold')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')

        ax.set_xlim(min(self.X)-Xrange, max(self.X)+Xrange)
        ax.set_ylim(min(self.Y)-Yrange, max(self.Y)+Yrange)
        # plt.show()

    def find_coeff_by_math(self):
        self.check_presence('r', self.Pearson)

        self.a = (self.r * self.std_y) / self.std_x
        self.b = self.mean_y - self.a * self.mean_x

        self.yhat = self.a * self.X + self.b
    
    def check_presence(self, arg, func):
        if arg not in self.__dict__:
            func()
    
    def plot_with_regression(self):
        self.check_presence('yhat', self.find_coeff_by_math)
        
        self.plot_scatter()
        plt.plot(self.X, self.yhat, color='red', label='y = {}x + {}'.format(*map(lambda x: round(x,4), (self.a, self.b))))
        # plt.axvline(0, color='black')
        # plt.axhline(0, color='black')
        plt.legend()

    def Pearson(self):
        self.check_presence('mean_x', self.X_Y_stats)
            
        self.mean_xy = np.mean(self.X * self.Y)
        self.r = (self.mean_xy - self.mean_x * self.mean_y) / (self.std_x * self.std_y)

        return self.r

    def cov_coeff(self):
        self.check_presence('mean_x', self.X_Y_stats)
        n = len(self.X)
        
        self.cov = sum((self.X - self.mean_x) * (self.Y - self.mean_y)) / n
        return self.cov
    
    def r2_coeff(self):
        self.check_presence('r', self.Pearson)

        self.r2 = self.Pearson_coeff**2
        return self.r2

    def X_Y_stats(self):
        self.mean_x, self.std_x = self.get_stats_data(self.X)
        self.mean_y, self.std_y = self.get_stats_data(self.Y)
    
    def get_stats_data(self, series):
        series = SB.discreetVariationSeries(series)
        return series.mean, series.std

    def approx(self, x):
        self.check_presence('a', self.find_coeff_by_math)

        return self.a * x + self.b