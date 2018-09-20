import numpy as np
from .certificates import Certificates


class Simplex:

    _eps = 1e-07

    _tableau = None
    _basic_solution = None

    _m = None
    _n = None

    def __init__(self, c, A, b, el_op):

        self.m = A.shape[0]
        self.n = A.shape[1]

        self.tableau = Simplex.build_tableau(c, A, b, el_op)

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
    def tableau(self):
        return self._tableau

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

    @tableau.setter
    def tableau(self, t):
        self._tableau = t

    @basic_solution.setter
    def basic_solution(self, bs):
        self._basic_solution = bs

    def eps_test(self, v1, v2):
        return abs(v1 - v2) <= self.eps

    def print_tableau(self):
        print("T =\n{}".format(self.tableau))

    def get_objective_value(self):
        return self.tableau[0][self.n]

    def get_solution(self):
        pass

    def canonical(self):

        t = self.tableau

        for i in range(1, self.m+1):
            t[0] -= t[i]

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
            if all([self.eps_test(x, 0) or x < 0 for x in t[1:, j]]) and t[0][j] < 0:
                self.certificate = Certificates.UNBOUNDED
                return False

        return True

    def run(self, canonize=False):

        if canonize:
            self.canonical()

        t = self.tableau

        while self.keep_going(t):
            i, j = self.choose_pivot(t)
            self.pivot(t, i, j)

        if self.certificate is Certificates.FEASIBLE:
            return self.tableau[0][-1]

        return self.tableau[0][self.m]

    @classmethod
    def build_tableau(cls, c, A, b, el_op):

        c = -c
        zero = np.zeros((1, 1))

        c = np.concatenate((c, zero), axis=1)
        linear_system = np.concatenate((A, b), axis=1)
        t = np.concatenate((c, linear_system))
        t = np.concatenate((t, el_op), axis=1)

        return t