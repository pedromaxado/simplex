import numpy as np
from tp_1.algorithms import Simplex


class LinearProgramming:

    _c = None
    _A = None
    _b = None

    def __init__(self, lp):

        self.c, self.A, self.b = LinearProgramming.standardize_lp(lp)

    @property
    def c(self):
        return self._c

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @c.setter
    def c(self, value):
        self._c = value

    @A.setter
    def A(self, value):
        self._A = value

    @b.setter
    def b(self, value):
        self._b = value

    def print_lp(self):

        print("c =\n{}".format(self.c))
        print("A =\n{}".format(self.A))
        print("b =\n{}".format(self.b))

    def solve(self):

        s = Simplex(self.c, self.A, self.b)
        s.run()
        s.print_tableau()

    @classmethod
    def standardize_lp(cls, lp):

        m = lp['m']
        n = lp['n']

        add_cols = n - np.count_nonzero(lp['vars_sign'])

        c = lp['c']
        A = lp['A']

        new_c = np.empty([1, n + add_cols])
        new_A = np.empty([m, n + add_cols])
        new_b = lp['b']

        for idx, x in enumerate(lp['vars_sign']):
            if x:
                new_A[:, idx] = A[:, idx]
                new_c[0][idx] = c[0][idx]
            else:
                new_A[:, idx] = A[:, idx]
                new_A[:, idx+1] = -A[:, idx]

                new_c[0][idx] = c[0][idx]
                new_c[0][idx+1] = -c[0][idx]

        for ln, sign in enumerate(lp['inequality_signs']):
            if sign is not 0:
                gap_var = np.zeros((m, 1))
                gap_var[ln][0] = sign

                new_c = np.concatenate((new_c, np.zeros((1, 1))), axis=1)
                new_A = np.concatenate((new_A, gap_var), axis=1)

        return new_c, new_A, new_b
