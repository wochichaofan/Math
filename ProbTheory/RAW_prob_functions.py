import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.stats import norm
from decimal import *
import pandas as pd
getcontext().prec = 4


class Stats_Functions:
    def __init__(self):
        self.d_func = {
            'M(X)' : exp_val,
            'D(X)' : dispersity,
            'std(X)' : std,
            'Prob std for M(X)' : prob_std
        }
    
    def exp_val(self, x: list, p: list):
        return round(np.sum([xi*pi for xi, pi in zip(x, p)]), 2)

    def dispersity(self, x: list, p: list):
        return round(exp_val(np.power(x, 2), p) - np.power(exp_val(x, p), 2), 4)

    def std(self, x: list, p: list):
        return round(np.sqrt(dispersity(x, p)), 4)

    def disp_diff(self, x, p):
        return round(np.sum((x - exp_val(x, p))**2 * p), 2)

    def C(self, m, n):
        fact = np.math.factorial
        return int(fact(n) / (fact(m) * fact(n - m)))

    def prob_std(self, x, p):
        M = Decimal(exp_val(x, p))
        stand = Decimal(std(x, p))
        x1, p1 = build_prob_func(x, p, printer=False)

        f1 = M - stand
        f2 = M + stand

        for xi, pi in zip(x1, p1):
            if xi[0] < f1 and f1 <= xi[-1]:
                f1 = pi[0]

            if xi[0] < f2 <= xi[-1]:
                f2 = pi[0]

        return f2 - f1
    
    def Bernoulli_prob(self, m, n, p, q):
        C = self.C(m, n)

        return float(C * Decimal(np.math.pow(p, m)) * Decimal(np.math.pow(q, n - m)))

    def Poisson_prob(self, m, n, p=None, lmbd=None):
        fact = np.math.factorial
        if p is None:
            return float(Decimal(np.math.pow(lmbd, m) / fact(m)) * Decimal(np.exp(-lmbd)))
        else:
            lmbd = n*p
            return float(Decimal(np.math.pow(lmbd, m) / fact(m)) * Decimal(np.exp(-lmbd)))
        
        
class Distribution_funcs(Stats_Functions):
    def __init__(self):
        super().__init__()
        
    def geometric_destribution(self, n, p):
        q = round(1-p, 4)

        d = {xi: round(q**(xi-1) * p, 4) for xi in range(1, n)}    
        d[n] = round(q**(n-1), 4)

        assert sum(d.values()) == 1, 'sum not equal to 1, check input data'
        return d

    def binominal_destribution(self, m, n, p):
        q = (1-Decimal(p))

        return {i : self.Bernoulli_prob(i, n, p, q) for i in range(m+1)}

    def Poisson_dist(n=None, p=None, lmbd=None):
        if lmbd == None:
            assert n*p >= 0, 'Poisson can\'t be used: lambda is less than 0'
            lmbd = n*p
            return {xi : self.Poisson_prob(xi, n, lmbd=lmbd) for xi in range(n)}
        else:
            return {xi : self.Poisson_prob(xi, n, lmbd=lmbd) for xi in range(n)}

    def hyper_geometry_dist(N, M, n, x_exp=None):
        """
        N - кол-во объектов в совокупности
        M - кол-во объектов искомого класса
        n - размер выборки
        x_exp - размер максимального ожидаемого кол-ва объектов в выборке
        """

        x_exp = n if x_exp is None or x_exp < n else x_exp
        x_exp = M if M < x_exp else x_exp

        d = {xi: round(Decimal((C(xi, M) * C(n-xi, N-M)) / C(n, N)), 4) for xi in range(0, x_exp+1)}
        return d