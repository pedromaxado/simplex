import numpy as np
from fractions import Fraction


def read_file(file_name):
    """Reads a file containing values that defines a linear programming.

    Args:
        file_name (string): The file name.

    Returns:
        list: A list built by removing the end of line characters and
        splitting the lines by spaces.

    """

    with open(file_name, 'r') as f:
        lp = f.readlines()

    for i in range(len(lp)):
        lp[i] = lp[i].rstrip().split(' ')

    return lp


def build_lp(lp):
    """Method that builds a dict of the linear programming information
    based on the pattern defined in the specs.

    Args:
        lp (list): A bizarre list read from a file.

    Returns:
        dict: Dictionary with the linear's programming objective function
        and constraints.

    """

    n = int(lp[0][0])
    m = int(lp[1][0])

    lp_dict = {
        "n": n,
        "m": m,

        "vars_sign": [int(x) for x in lp[2]],

        "c": np.array([[Fraction(x) for x in lp[3]]], dtype=Fraction),
        "A": np.array([[Fraction(lp[i][j]) for j in range(n)] for i in range(4, 4 + m)], dtype=Fraction),
        "b": np.array([[Fraction(lp[i][-1]) for i in range(4, 4 + m)]], dtype=Fraction).transpose(),

        "inequality_signs": [1 if lp[i][-2] == "<=" else -1 if lp[i][-2] == ">=" else 0 for i in range(4, 4 + m)]
    }

    return lp_dict


def get_lp_from_file(file_name):
    return build_lp(read_file('../data/pls/' + file_name))


def write_result_on_file(file_name, msg_str):

    with open('../data/solutions/' + file_name, 'w') as f:
        f.write(msg_str)


if __name__ == '__main__':
    print(get_lp_from_file('lp1.txt'))
