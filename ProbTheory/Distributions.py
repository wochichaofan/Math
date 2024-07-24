


class Distribution_funcs:
    def geometric_destribution(n, p):
        q = round(1-p, 4)

        d = {xi: round(q**(xi-1) * p, 4) for xi in range(1, n)}    
        d[n] = round(q**(n-1), 4)

        assert sum(d.values()) == 1, 'sum not equal to 1, check input data'
        return d

    def binominal_destribution(m, n, p):
        q = (1-Decimal(p))

        return {i : Bernoulli(i, n, p, q) for i in range(m+1)}

    def Poisson_dist(n=None, p=None, lmbd=None):
        if lmbd == None:
            assert n*p >= 0, 'Poisson can\'t be used: lambda is less than 0'
            lmbd = n*p
            return {xi : Poisson(xi, n, lmbd=lmbd) for xi in range(n)}
        else:
            return {xi : Poisson(xi, n, lmbd=lmbd) for xi in range(n)}

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