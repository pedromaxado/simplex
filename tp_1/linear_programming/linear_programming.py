import numpy as np
from tp_1.algorithms import Simplex
from tp_1.algorithms import Certificates


class LinearProgramming:

    _c = None
    _A = None
    _b = None

    _el_op = None

    def __init__(self, lp):

        self.c, self.A, self.b = LinearProgramming.build_lp(lp)

        self.el_op = np.eye(self.A.shape[0]+1, self.A.shape[0], k=-1, dtype=float)

    @property
    def c(self):
        return self._c

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def el_op(self):
        return self._el_op

    @c.setter
    def c(self, value):
        self._c = value

    @A.setter
    def A(self, value):
        self._A = value

    @b.setter
    def b(self, value):
        self._b = value

    @el_op.setter
    def el_op(self, value):
        self._el_op = value

    def print_lp(self):

        print("c =\n{}".format(self.c))
        print("A =\n{}".format(self.A))
        print("b =\n{}".format(self.b))

    def build_aux_lp(self):

        m = self.A.shape[0]
        n = self.A.shape[1]

        aux_c = np.concatenate((np.zeros((1, n)), np.full((1, m), -1)), axis=1)
        aux_A = np.copy(self.A)
        aux_b = np.copy(self.b)

        for i in range(m):
            if aux_b[i] < 0:
                aux_A[i] *= -1
                aux_b[i] *= -1

        aux_A = np.concatenate((aux_A, np.identity(m)), axis=1)

        return aux_c, aux_A, aux_b

    def solve(self):

        aux_c, aux_A, aux_b = self.build_aux_lp()

        s_aux = Simplex(aux_c, aux_A, aux_b, self.el_op)
        s_aux.run(canonize=True)
        s_aux.print_tableau()

        if s_aux.eps_test(s_aux.tableau[0][s_aux.n], 0):
            self.A = s_aux.tableau[1:, :s_aux.n-s_aux.m]
            self.b = np.array([s_aux.tableau[1:, s_aux.n]]).T
            self.el_op = s_aux.tableau[:, aux_A.shape[1]+1:]

            s = Simplex(self.c, self.A, self.b, self.el_op)
            s.run(canonize=True)
            s.print_tableau()
        else:
            print("inviavel")

    @classmethod
    def build_lp(cls, lp):

        m = lp['m']
        n = lp['n']

        add_cols = n - np.count_nonzero(lp['vars_sign'])

        c = lp['c']
        A = lp['A']

        new_c = np.zeros([1, n + add_cols])
        new_A = np.zeros([m, n + add_cols])
        new_b = lp['b']

        col = 0
        a_idx = 0
        while col < n + add_cols:
            sign = lp['vars_sign'][a_idx]

            new_A[:, col] = A[:, a_idx]
            new_c[0][col] = c[0][a_idx]

            if not sign:
                col += 1
                new_A[:, col] = -A[:, a_idx]
                new_c[0][col] = -c[0][a_idx]

            col += 1
            a_idx += 1

        for ln, sign in enumerate(lp['inequality_signs']):
            if sign is not 0:
                gap_var = np.zeros((m, 1))
                gap_var[ln][0] = sign

                new_c = np.concatenate((new_c, np.zeros((1, 1))), axis=1)
                new_A = np.concatenate((new_A, gap_var), axis=1)

        return new_c, new_A, new_b
