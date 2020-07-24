
from .atomic_states import np

EPS = 1e-5
class gauss_mode:

    """
    Define a solution of 'paraxial' wave equation - Gauss mode TEM_{00}
    """

    def __init__(self, waist, omega):

        self.waist = waist
        self.omega = omega
        self.z0 = np.pi * self.waist ** 2 * self.omega / (2*np.pi)  # Rayleigh length

        c = 1
        k = self.omega / c

        if self.waist >= 5.*self.omega:
            print('Using gaussian approximation...')
            self.forward_amplitude = \
            lambda r, z: np.sqrt(2 / np.pi) / self.evo_waist(z) * np.exp(-1 * r ** 2 / self.evo_waist(z) ** 2) \
                         * np.exp(+1j * k * (z + r**2*self.curvature2(z)) - 1j * self.gouy_phase(z))

            self.backward_amplitude = \
            lambda r, z: np.sqrt(2 / np.pi) / self.evo_waist(z) * np.exp(-1 * r ** 2 / self.evo_waist(z) ** 2) \
                         * np.exp(-1j * k * (z + r**2*self.curvature2(z)) + 1j * self.gouy_phase(z))
        else:
            self.forward_amplitude = lambda r,z: self.exact_mode(r,z)
            self.backward_amplitude = lambda r,z: np.conj(self.exact_mode(r,z))
    """
    Complex waist of beam as a function of position:
    """

    def evo_waist(self, z):
        #return self.waist
        #Future
        #curvature radius
        return self.waist * np.sqrt(1+(z/self.z0)**2)

    """
    Complex multiplicator (for small waists):
    """
    def complex_waist(self,z):
        return 1 - 0*(4 / self.waist ** 2 / self.omega ** 2) * 1/(1-1j*z/self.z0)

    def curvature2(self, z):
        #return 0
        #Future
        return 1 / (2 * (z+10e-4) * (1+(z/self.z0)**2))

    def gouy_phase(self, z):
        #return 0
        #Future
        return np.arctan(z/self.z0)

    def norm(self,z):
        return 1

    def exact_mode(self, r, z):

        from scipy.integrate import quad
        from scipy.special import j0,j1

        self.ez = np.empty_like(z, dtype=np.complex64)

        res = np.empty_like(z, dtype=np.complex64)
        for i in range(len(z)):
            re = lambda x: x*np.exp(-self.waist ** 2 * x ** 2 / 4) * np.cos(
                np.sqrt(self.omega ** 2 - x ** 2) * z[i]) * j0(x * r[i])
            im = lambda x: x*np.exp(-self.waist ** 2 * x ** 2 / 4) * np.sin(
                np.sqrt(self.omega ** 2 - x ** 2) * z[i]) * j0(x * r[i])
            eva = lambda x: x*np.exp(-self.waist ** 2 * x ** 2 / 4 -
                np.sqrt(x ** 2 - self.omega ** 2) * abs( z[i] )) * j0(x * r[i])

            rez = lambda x: x*np.exp(-self.waist ** 2 * x ** 2 / 4) * np.cos(
                np.sqrt(self.omega ** 2 - x ** 2) * z[i]) * j1(x * r[i]) * x/np.sqrt(self.omega**2 - x**2)
            imz = lambda x: x*np.exp(-self.waist ** 2 * x ** 2 / 4) * np.sin(
                np.sqrt(self.omega ** 2 - x ** 2) * z[i]) * j1(x * r[i]) * x/np.sqrt(self.omega**2 - x**2)

            norm_prop = lambda x: x*self.waist ** 2 / 2 * (1 + self.omega**2 / (self.omega ** 2 - x**2)) \
                                                          * np.exp(-self.waist**2 * x**2 / 2)

            norm_eva = lambda x: x*self.waist ** 2 / 2 * (1 + self.omega**2 / (self.omega ** 2 - x**2)) \
                                                         * np.exp(-self.waist**2 * x**2 / 2) \
                                 * (np.exp(-2*np.sqrt(x**2 - self.omega**2)*abs(z[i]))-1)



            om = (1-EPS)*self.omega
            op = (1+EPS)*self.omega
            norm = quad(norm_prop, 0, om)[0] + \
                   0*quad(norm_prop, op, np.inf)[0] + \
                   0*quad(norm_eva, op, np.inf)[0]

            res[i] = (quad(re, 0, self.omega)[0] +
                      1j*quad(im, 0, self.omega)[0] +
                      0*quad(eva, self.omega, np.inf)[0])*\
                     self.waist / np.sqrt(2*np.pi) / np.sqrt(norm)

            self.ez[i] = -1j*(quad(rez, 0, self.omega)[0] +
                      1j*quad(imz, 0, self.omega)[0] +
                      0*quad(eva, self.omega, np.inf)[0])*\
                     self.waist / np.sqrt(2*np.pi) / np.sqrt(norm)

        return res

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    om = 1
    w = 0.1*np.pi

    m = GaussMode(w, om)
    rho = np.linspace(-10,10, 100)
    amps = abs(m.exact_mode(rho, 0*np.ones_like(rho)))**2
    amsg = abs(m.forward_amplitude(rho, 0*np.ones_like(rho)))**2
    amsz = abs(m.ez)**2

    ampsR = abs(m.exact_mode(rho, m.z0*np.ones_like(rho)))**2
    amsgR = abs(m.forward_amplitude(rho, m.z0*np.ones_like(rho)))**2
    amszR = abs(m.ez) ** 2
    plt.plot(rho, amps, 'b-')
    plt.plot(rho, amsg, 'r-')
    plt.plot(rho, amsz, 'g-')

    plt.plot(rho, ampsR, 'b--')
    plt.plot(rho, amsgR, 'r--')
    plt.plot(rho, amszR, 'g--')
    plt.show()

    #print(m.exact_mode(np.zeros_like(x),x))






