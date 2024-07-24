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