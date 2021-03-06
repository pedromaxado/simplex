import numpy as np
from tp_1.linear_programming import LinearProgramming
from tp_1.utils import get_lp_from_file, write_result_on_file


def run():

    np.set_printoptions(precision=4, suppress=True, linewidth=500)

    for i in range(0, 10):
        lp = LinearProgramming(get_lp_from_file('pl_y' + str(i) + '.txt'))
        message = lp.solve()

        write_result_on_file('pl_y' + str(i) + '.txt', message)

    file_prefix = 'inviavel'

    lp = LinearProgramming(get_lp_from_file(file_prefix + '.txt'))
    message = lp.solve()

    write_result_on_file(file_prefix + '.txt', message)


if __name__ == '__main__':
    run()
