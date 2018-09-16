import numpy as np
from tp_1.algorithms import Simplex


def run():
    c = np.array([[2, 3, 0, 0, 0]], dtype=float)

    A = np.array([[1, 1, 1, 0, 0],
                  [2, 1, 0, 1, 0],
                  [-1, 1, 0, 0, 1]], dtype=float)

    b = np.array([[6, 10, 4]], dtype=float)

    s = Simplex(c, A, b.transpose())
    s.run()

    print(s.t)


if __name__ == '__main__':
    run()
