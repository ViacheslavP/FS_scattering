from scipy.sparse import lil_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csr_matrix
from .utills import counter
from .atomic_states import np
from .atomic_states import atomic_state

"""
Solvers:
1. w/o dispersion and limited basis
2. w/o dispersion  and extended basis
"""

# Initializion

hbar = 1  # dirac's constant = 1 => enrergy = omega
c = 1  # speed of light c = 1 => time in cm => cm in wavelenght
gd = 1.  # vacuum decay rate The only
lambd = 1  # atomic wavelength (lambdabar)
k = 1 / lambd

# dipole moments (Wigner-Eckart th.), atom properties

d01 = np.sqrt(hbar * gd / 4 * (lambd ** 3))  # |F = 0, n = 0> <- |F0 = 1> - populated level
d10 = d01
d00 = np.sqrt(hbar * gd / 4 * lambd ** 3)
d01m = np.sqrt(hbar * gd / 4 * lambd ** 3)
d1m0 = d01m

# Validation (all = 1 iff ful theory, except SINGLE_RAMAN)

DDI = 1
VACUUM_DECAY = 0  # = 0 iff assuming only decay into fundamental mode, =1 iff decay into fundamental and into radiation
PARAXIAL = 1  # = 0 iff paraxial, =1 iff full mode
SINGLE_RAMAN = True
FIX_RANDOM = True
SEED = 5
RABI = 1e-16  # 4#4+1e-16 #Rabi frequency
DC = 0  # Rabi detuning
SHIFT = 0

FULL = 0
FIRST = 1  # = 0 iff assuming full subtraction
SECOND = 1

EPS = (2 * np.pi * 6.065) / (384 * 1e6)  # \gamma / \omega
from .gmode import gauss_mode
#Signal mode
waist_signal = 2*np.pi * 1400 / 780 * lambd
sm = gauss_mode(waist_signal, 1.)
VG = 1
VPH = 1
ZR = sm.z0
Ef = lambda x,y,z: sm.forward_amplitude(np.sqrt(x**2 + y**2), z)
Eb = lambda x,y,z: sm.backward_amplitude(np.sqrt(x**2 + y**2), z)

#Trap mode
waist_trap = 2*np.pi * 1400 / 780 * lambd
trap_scale = 0.5 * np.sqrt(10 / 1000) # from Boltzmann distribution sqrt(kT / U_0)
tm = gauss_mode(waist_trap, 1.)
sxy = waist_trap * trap_scale
sz = tm.z0 * trap_scale


def sigma_matrix(array: object):
    pos = array.pos
    noa = array.noa

    rpos = np.asarray([[sum((pos[:, i] - pos[:, j])**2) for i in range(noa)] for j in range(noa)])
    rpos_ef = rpos + np.eye(noa)
    xm = np.asarray([[(pos[0, i] - pos[0, j] - 1j*(pos[1,i] - pos[1,j]))/rpos_ef[i,j] for i in range(noa)] for j in range(noa)])
    xp = -np.asarray([[(pos[0, i] - pos[0, j] + 1j*(pos[1,i] - pos[1,j]))/rpos_ef[i,j] for i in range(noa)] for j in range(noa)])
    x0 = np.asarray([[(pos[2, i] - pos[2, j])/rpos_ef[i,j] for i in range(noa)] for j in range(noa)])

    D1 = ((np.ones_like(rpos) - 1j * rpos - rpos ** 2) / (rpos_ef ** 3) * np.exp(1j * rpos)) \
         * (np.ones(noa) - np.eye(noa))
    D2 = -1 * ((DDI * 3 - 3 * 1j * rpos - rpos ** 2) / (rpos_ef ** 3) * np.exp(1j * rpos)) \
         * (np.ones(noa) - np.eye(noa))

    sigmai = np.zeros([noa, noa, 3, 3], dtype=np.complex)

    sigmai[:, :, 0, 0] = d01m * d1m0 * (D1 - xp * xm * D2)
    sigmai[:, :, 0, 1] = d01m * d00 * (-1 * xp * x0 * D2)
    sigmai[:, :, 0, 2] = d01m * d10 * (-xp * xp * D2)
    sigmai[:, :, 1, 0] = d00 * d1m0 * (x0 * xm * D2)
    sigmai[:, :, 1, 1] = d00 * d00 * (D1 + x0 * x0 * D2)
    sigmai[:, :, 1, 2] = d00 * d10 * (x0 * xp * D2)
    sigmai[:, :, 2, 0] = d01 * d1m0 * (-xm * xm * D2)
    sigmai[:, :, 2, 1] = d01 * d00 * (-xm * x0 * D2)
    sigmai[:, :, 2, 2] = d01 * d10 * (D1 - xm * xp * D2)

    return sigmai

def super_matrix_full(array: object):

    nat = array.noa
    nb = nat - 1
    from itertools import product as combi
    state = np.asarray([i for i in combi(range(3), repeat=nb)])
    st = np.ones([nat, nat, 3 ** nb], dtype=int) * 2
    index = np.asarray(np.argsort(array.pos)[:, 1:nb + 1], dtype=np.int)
    for n1 in range(nat):
        k = 0
        for n2 in np.sort(index[n1, :]):
            for i in range(3 ** nb):
                st[n1, n2, i] = state[i, k]
            k += 1

    sigma = lil_matrix([3 ** nb * nat, 3 ** nb * nat], dtype=np.complex)
    sigmai = sigma_matrix(array)

    def foo(ni1, nj2, i, j):
        for ki1 in np.append(index[ni1, :], index[nj2, :]):
            if ki1 == ni1 or ki1 == nj2: continue;
            if st[ni1, ki1, i] != st[nj2, ki1, j]:
                return False
        return True

    for n1 in range(nat):
        for n2 in range(nat):  # choose initial excited atom
            for i in range(3 ** nb):  # select initial excited atom environment
                # choose final excited or np.sort(self.index[n1,:]): if massive photon

                for j in range(3 ** nb):  # select final excited environment
                    if foo(n1, n2, i, j):  # if transition is possible then make assigment
                        sigma[n1 * 3 ** nb + i, n2 * 3 ** nb + j] = \
                            sigmai[n1, n2, st[n1, n2, i], st[n2, n1, j]]
    return csr_matrix(sigma)

def super_matrix_ext(chain: object, omega=0):
    dm = sigma_matrix(chain)
    noa = chain.noa
    dim = noa + 2 * noa * (noa - 1)

    sm = lil_matrix((dim, dim), dtype=np.complex)
    for initial in range(dim):
        sm[initial, initial] = 1.
        if initial < noa:  # If initial state has no raman atoms (i.e. |+++e+++>)
            for s in range(noa - 1):  # s is final excited atom (rel. position)

                sr = noa + 2 * (noa - 1) * initial + 2 * s

                ne = s
                if s >= initial:
                    ne += 1  # ne is excited atom position (true)

                sm[initial, ne] = dm[initial, ne, 2, 2]  # Transition to rayleigh
                sm[initial, sr] = dm[initial, ne, 2, 1]  # Transition to Raman with m=0
                sm[initial, sr + 1] = dm[initial, ne, 2, 0]
                # assert initial != sr
                # assert initial != sr+1
        elif initial >= noa:
            k = (initial - noa) // (2 * (noa - 1))  # Raman atom
            ni = (initial - noa) % (2 * (noa - 1)) // 2  # Excited atom rel pos
            gk = (initial - noa) % (2 * (noa - 1)) % 2  # Raman state (0 or 1)
            assert initial == noa + 2 * (noa - 1) * k + 2 * ni + gk

            if ni >= k:
                ni += 1

            for s in range(noa - 1):
                ne = s
                if s >= k:
                    ne += 1

                # w/ same raman atom
                sr = noa + 2 * (noa - 1) * k + 2 * s + gk
                if sr != initial:
                    sm[initial, sr] = dm[ni, ne, 2, 2]

            # transfer exc to the Raman atom
            if ni < k:
                s = k - 1
            else:
                s = k

            sr = noa + 2 * (noa - 1) * ni + 2 * s
            assert initial != sr
            assert initial != sr + 1

            sm[initial, k] = dm[ni, k, gk, 2]  # The only way to transfer back to elastics
            sm[initial, sr] = dm[ni, k, gk, 0]
            sm[initial, sr + 1] = dm[ni, k, gk, 1]
            continue
    return csr_matrix(sm)


def solver4(array: object, freqs, rabi=0, dc=0.00001):
    noa = array.noa
    dim = len(array.campl)
    nof = len(freqs)
    instate = array.campl
    scV = np.empty((dim, nof), dtype=np.complex)
    oned = csr_matrix(eye(dim, dim, format='csr', dtype=np.complex))
    sigma = super_matrix_ext(array)
    for i,om in enumerate(freqs):
        omg = -1 - om + rabi ** 2 / (4 * (om - dc)) - 0.5j
        resolvent = omg  * oned + sigma
        scV[:, i] = spsolve(resolvent, instate)
        exitCode = 0
        try:
            assert exitCode == 0
        except AssertionError:
            if exitCode > 0:
                print(f'Convergence not achieved. Step {i}, Number of iterations {exitCode} \n Continue ...')
            elif exitCode < 0:
                print('Something bad happened. Exitting...')
                assert exitCode == 0
        counter(i, nof)

    return scV

def solver4_step(array: object, om, rabi=0, dc=0.00001):
    noa = array.noa
    dim = len(array.campl)
    instate = array.campl

    scV = np.empty((dim), dtype=np.complex)
    oned = eye(dim, dim, format='csr', dtype=np.complex)

    Sigma = super_matrix_full(array)
    omg = -om + rabi ** 2 / (4 * (om - dc)) - 0.5j
    resolvent = omg * oned + Sigma
    scV = spsolve(resolvent, instate)

    return scV
