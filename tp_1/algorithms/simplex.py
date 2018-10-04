import numpy as np
from .certificates import Certificates


class Simplex:

    _eps = 1e-09

    _tableau = None
    _base = None
    _unbounded_col = None

    _m = None
    _n = None

    def __init__(self, c, A, b, el_op, base=None):

        self.m = A.shape[0]
        self.n = A.shape[1]

        self.tableau = Simplex.build_tableau(c, A, b, el_op)

        if base is not None:
            self.base = base
        else:
            self.base = np.full([self.n], -1)
            self.base[self.n - self.m:] = list(range(1, self.m + 1))

        self.unbounded_col = -1

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
    def base(self):
        return self._base

    @property
    def unbounded_col(self):
        return self._unbounded_col

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

    @base.setter
    def base(self, bs):
        self._base = bs

    @unbounded_col.setter
    def unbounded_col(self, value):
        self._unbounded_col = value

    def eps_test(self, v1, v2):
        return abs(v1 - v2) <= self.eps

    def print_tableau(self):
        print("T =\n{}".format(self.tableau))

    def get_obj_value(self):
        return self.tableau[0][self.n]

    def get_solution(self):

        solution = []

        for i in range(self.n):
            bi = self.base[i]
            if bi == -1:
                solution.append(0)
            else:
                solution.append(self.tableau[bi][self.n])

        return solution

    def get_certificate(self):

        if self.certificate in [Certificates.FEASIBLE, Certificates.INFEASIBLE]:
            return self.tableau[0, self.n+1:].tolist()

        certificate = np.zeros(self.n)
        certificate[self.unbounded_col] = 1
        col_a = 1

        for i in [x for x in range(self.n) if x != self.unbounded_col]:
            if self.base[i] != -1:
                certificate[i] = -self.tableau[col_a][self.unbounded_col]
                col_a += 1
            else:
                certificate[i] = 0

        return certificate

    def get_c(self):
        return self.tableau[0, :self.n]

    def get_A(self):
        return self.tableau[1:, :self.n - self.m]

    def get_b(self):
        return np.array([self.tableau[1:, self.n]]).T

    def canonical(self):

        t = self.tableau

        for i in range(self.n):
            if self.base[i] > -1:
                t[0] -= t[0][i]*t[self.base[i]]

    def update_base(self, i, j):

        for idx in range(self.n):
            if self.base[idx] == i:
                self.base[idx] = -1

        self.base[j] = i

    def pivot(self, t, i, j):

        t[i] = (1/t[i][j])*t[i]

        for ln in [x for x in range(self.m+1) if x != i]:
            if not self.eps_test(t[ln][j], 0):
                t[ln] = (-t[ln][j])*t[i] + t[ln]

        self.update_base(i, j)

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
                self.unbounded_col = j
                return False

        return True

    def run(self, canonize=False):

        if canonize:
            self.canonical()

        t = self.tableau

        while self.keep_going(t):
            i, j = self.choose_pivot(t)
            self.pivot(t, i, j)

        return {
            "obj_value": self.get_obj_value(),
            "base": self.base[:self.n-self.m],
            "solution": self.get_solution(),
            "feasibility": self.certificate,
            "certificate": self.get_certificate()
        }

    @classmethod
    def build_tableau(cls, c, A, b, el_op):

        c = -c
        zero = np.zeros((1, 1))

        c = np.concatenate((c, zero), axis=1)
        linear_system = np.concatenate((A, b), axis=1)
        t = np.concatenate((c, linear_system))
        t = np.concatenate((t, el_op), axis=1)

        return t
