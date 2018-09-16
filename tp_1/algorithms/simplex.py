import numpy as np
from .certificates import Certificates


class Simplex:

    zero = np.zeros((1, 1))
    _eps = 1e-07

    _t = None
    _basic_solution = None

    _m = None
    _n = None

    def __init__(self, c, A, b):

        self.m = A.shape[0]
        self.n = A.shape[1]

        self.t = Simplex.build_tableau(c, A, b, A.shape[0])

        self.certificate = Certificates.FEASIBLE

    @property
    def eps(self):
        return self._eps

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    @property
    def t(self):
        return self._t

    @property
    def basic_solution(self):
        return self._basic_solution

    @eps.setter
    def eps(self, eps):

        if eps >= 0:
            self._eps = eps
        else:
            raise Exception("Epsilon must be non negative!")

    @m.setter
    def m(self, m):
        self._m = m

    @n.setter
    def n(self, n):
        self._n = n

    @t.setter
    def t(self, t):
        self._t = t

    @basic_solution.setter
    def basic_solution(self, bs):
        self._basic_solution = bs

    def print_tableau(self):
        print("T =\n{}".format(self.t))

    def eps_test(self, v1, v2):
        return abs(v1 - v2) <= self.eps

    @classmethod
    def build_tableau(cls, c, A, b, m):

        c = -c
        zero = np.zeros((1, 1))

        id = np.eye(m+1, m, k=-1, dtype=float)

        c = np.concatenate((c, zero), axis=1)
        linear_system = np.concatenate((A, b), axis=1)
        t = np.concatenate((c, linear_system))
        t = np.concatenate((t, id), axis=1)

        return t

    def pivot(self, t, i, j):

        t[i] = (1/t[i][j])*t[i]

        for ln in [x for x in range(self.m+1) if x != i]:
            if not self.eps_test(t[ln][j], 0):
                t[ln] = (-t[ln][j])*t[i] + t[ln]

    def choose_pivot(self, t):

        col = np.argmin(t[0, :self.n])
        ratios = []

        for i in range(1, self.m + 1):
            if not self.eps_test(t[i][col], 0) and not t[i][col] < 0:
                ratios.append([t[i][self.n]/t[i][col], i])

        return min(ratios)[1], col

    def keep_going(self, t):

        if all([self.eps_test(x, 0) or x > 0 for x in t[0, :self.n]]):
            return False

        for j in range(self.n + 1):
            if all([self.eps_test(x, 0) or x < 0 for x in t[:, j]]):
                self.certificate = Certificates.UNBOUNDED
                return False

        return True

    def run(self):

        t = self.t

        while self.keep_going(t):
            i, j = self.choose_pivot(t)
            self.pivot(t, i, j)

        if self.certificate is Certificates.FEASIBLE:
            return self.t[0][-1]
