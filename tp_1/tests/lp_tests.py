import numpy as np
from tp_1.linear_programming import LinearProgramming
from tp_1.utils import get_lp_from_file


def run():

    np.set_printoptions(precision=4, suppress=True, linewidth=500)

    lp = LinearProgramming(get_lp_from_file('inviavel.txt'))
    #lp.print_lp()
    lp.solve()


if __name__ == '__main__':
    run()
