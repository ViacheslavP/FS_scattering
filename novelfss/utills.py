from .atomic_states import np
import sys

CSVPATH = 'data/fig_def/'


def counter(i, n):
    ist = 100 * (i + 1) / n
    sys.stdout.write("\r%d%%" % ist)
    sys.stdout.flush()
    sys.stdout.write("\033[K")


def to_mathematica(fname, *argv, csvpath=CSVPATH):
    import os
    if not os.path.exists(csvpath):
        os.makedirs(csvpath)
    toCsv = np.column_stack(argv)
    np.savetxt(csvpath + fname + '.csv', toCsv, delimiter=',', fmt='%1.8f')
