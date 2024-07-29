from scipy.stats import norm
from scipy.stats import t, chi2

stat_round = lambda x: round(x, 4)

def Laplace_table(x):
    return round(norm.cdf(x)-.5, 4)

def Laplace_table_inverse(x):
    return stat_round(norm.ppf(x/2 + .5))

def tinv(alpha, n):
    """
    Student\'s confidence coefficient
    """
    return stat_round(t.isf((1-alpha)/2, n-1))

def chiinv(prob, n):
    return chi2.isf(prob, n-1)