import numpy as np

# methods and procedures for creating various chains
class atomic_state(object):
    def __init__(self, pos, campl):

        self.noa = pos.shape[1]
        self._mpos = np.asarray(pos, dtype=np.float64)
        self.dim = self.noa + 2 * self.noa * (self.noa - 1)

        #not shuffled; run atomic_state.shuffle() for shuffling
        self.pos = self._mpos

        if not np.vdot(campl, campl) == 0:
            self.campl = campl / np.vdot(campl, campl)
        else:
            self.campl = campl

        if pos.shape[0] != 3:
            raise TypeError("Something wrong with the atomic state")

    def __add__(self, other):
        return self.merge_atomic_states(other, distance=0)

    def shuffle(self):
        self.pos = self._mpos
        from .dyson_solvers import sxy, sz
        self.pos[0] += np.random.normal(0.0, sxy, self.noa)
        self.pos[1] += np.random.normal(0.0, sxy, self.noa)
        self.pos[2] += np.random.normal(0.0, sz, self.noa)

    def illuminate(self, V):
        amps = V(*self.pos)
        dim = self.noa + 2 * self.noa * (self.noa - 1)
        self.campl = np.zeros(dim, dtype=np.complex)
        for i in range(self.noa):
            self.campl[i] = amps[i]


    def merge_atomic_states(self, bstate: object, distance=0) -> object:
        n1, n2 = self.noa, bstate.noa
        add_dist = 0
        zpos = np.concatenate((self._mpos, bstate._mpos + distance + add_dist), axis=None)
        campl = np.concatenate((self.campl, np.exp(2j * (distance + add_dist) * np.pi) * bstate.campl), axis=None)
        return atomic_state(zpos, campl)


# Create an atomic cloud around center
def new_cloud(noa:int):
    xpos = np.zeros(noa)
    ypos = np.zeros(noa)
    zpos = np.zeros(noa)

    poss = np.stack((xpos, ypos, zpos), axis=0)
    campl = np.ones_like(zpos, dtype=np.complex)

    return atomic_state(poss, campl)


# Create a simple chain of noa atoms with period d
def new_chain(noa: int, d, random=False):
    xpos = np.zeros(noa)
    ypos = np.zeros(noa)
    zpos = d * (np.arange(noa) - (noa-1)/2.)

    poss = np.stack((xpos, ypos, zpos), axis=0)
    campl = np.ones_like(zpos, dtype=np.complex)

    return atomic_state(poss, campl)

