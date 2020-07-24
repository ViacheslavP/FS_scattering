
from .wave_pack import convolution2
from .utills import counter
from .atomic_states import np


def get_psi_temporal(psi0: object, time, nof, f_max, solver='3'):
    noa = len(psi0.zpos)
    if solver=='4':
        from .dyson_solvers import solver4_step as solver
        dim = noa + 2*noa*(noa-1)
    elif solver=='3':
        from .dyson_solvers import solver3_step as solver
        dim = noa
    elif solver=='2':
        pass
    elif solver=='1':
        pass
    else:
        raise ValueError("Solver hasn't been found")

    freq = np.linspace(-f_max, f_max, nof)
    decay = np.zeros((dim, nof), dtype=np.complex)
    for i, om in enumerate(freq):
        decay[:, i] = solver(psi0, om)
        counter(i, nof)

    decay_temp = np.zeros((dim, len(time)), dtype=np.complex)

    for i in range(dim):
        decay_temp[i, :] = convolution2(time, freq, decay[i, :], np.ones_like(freq))
        counter(i,dim)

    return time, decay_temp


def get_temporal_metrics(setup: object, time2, nof, f_max, solver='3'):
    t, setup_td = get_psi_temporal(setup, time2, nof, f_max, solver)
    noa = len(setup.zpos)
    if solver=='4':
        instate = np.pad(setup.campl, (0, 2 * noa * (noa - 1)), 'constant', constant_values=(0, 0))
    elif solver == '3':
        instate = setup.campl
    setup_square = np.real(np.dot(np.conj(np.transpose(setup_td)), setup_td).diagonal())
    setup_init = abs(np.dot(np.conj(np.transpose(setup_td)), instate)) ** 2
    proj = np.diag(abs(instate))
    setup_p = np.dot(proj, setup_td)
    setup_inplace = np.real(np.dot(np.conj(np.transpose(setup_p)), setup_p).diagonal())
    return setup_square, setup_init, setup_inplace  # p, p0, pa


def get_pulses(setup: object, time2, nof, f_max, solver='3'):
    t, setup_td = get_psi_temporal(setup, time2, nof, f_max, solver)
    noa = len(setup.zpos)
    if solver=='3':
        leftstate = np.exp(-2j*np.pi* setup.zpos)
        rightstate = np.exp(2j*np.pi* setup.zpos)

        leftpulse = abs(np.dot(np.conj(np.transpose(setup_td)), leftstate)) ** 2
        rightpulse = abs(np.dot(np.conj(np.transpose(setup_td)), rightstate)) ** 2
    else:
        raise ValueError("Solver hasn't been realized yet")
    denominator = (leftpulse.max() + rightpulse.max())/2
    return leftpulse/denominator, rightpulse/denominator
