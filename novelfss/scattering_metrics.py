from .atomic_states import np
from .dyson_solvers import solver4
from .dyson_solvers import Ef, Eb

def absorbtion(array: object, freqs, rabi=0., rabi_dc=0.00001):
    array.illuminate(Ef)
    res = solver4(array, freqs, rabi, rabi_dc)
    transmission = 1 - 1j*2*np.pi*np.dot(np.conj(array.campl), res)

    return abs(transmission)**2 - np.ones_like(freqs)


def mean_absorbtion(array: object, freqs, nreps=100, rabi=0., rabi_dc=0.00001):
    absorb = np.zeros_like(freqs)
    for i in range(nreps):
        array.shuffle()
        absorb += absorbtion(array, freqs, rabi, rabi_dc) / nreps

    return absorb
