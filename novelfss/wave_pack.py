from .atomic_states import np
from scipy.integrate import simps
"""
Next 3 functions are: Fourier Integral Transform (FIT), reverse FIT, 
and convolution transformation in this particular task
"""

def FIT(array, freq):
    domega = freq[1]-freq[0]
    FIT_array = np.fft.fft(array)
    time = np.fft.fftfreq(array.size)*2*np.pi/domega
    FIT_array = np.multiply(FIT_array, domega * np.exp(-1j * (freq[0]+freq[-1])/2 * time) / (2 * np.pi))

    return time, FIT_array



def revFIT(FIT_array, time):
    dt = time[1]-time[0]
    array = np.fft.rfft(FIT_array)
    freq = np.fft.rfftfreq(FIT_array.size)*2*np.pi/dt
    array *= dt * np.exp(-complex(0, 1) * freq * time[0])
    return freq, array

def convolution(freq_c, kernel, pulse, vg=0.7):
    im  = np.multiply(kernel, pulse)
    time, fim = FIT(im,freq_c)
    _sw = np.argsort(time)
    time = time[_sw]
    fim = fim[_sw]
    return time, fim


def convolution2(time, freq_c, kernel, pulse):
    #Sticks and stones
    im = np.multiply(kernel, pulse)
    domega = freq_c[1] - freq_c[0]
    res = np.zeros_like(time, dtype=np.complex)
    for i, ti in enumerate(time):
        integrand = np.multiply(np.exp(-1j*ti*freq_c), im)
        res[i] = np.trapz(np.real(integrand), freq_c, domega) + 1j * np.trapz(np.imag(integrand), freq_c, domega)

    return res




def convolution_pad(freq_c, kernel, pulse, extnumber = 20):
    im  = np.multiply(kernel, pulse)
    #extent freqs:
    df = abs(freq_c[0] - freq_c[1])
    nof = freq_c.size
    edge = (2 * extnumber - 1)*( nof // 2 )
    if nof % 2 == 1:
        raise ValueError("Odd number of frequencies: unexpected results might happen")
    freq_ext = np.fft.fftfreq(nof*2*extnumber) * df * nof*2*extnumber
    _sf = np.argsort(freq_ext)
    freq_ext = freq_ext[_sf]

    im_pad = np.pad(im, (edge, edge), 'constant', constant_values=(0,0))
    #im_pad = im_pad - im_pad[-1]

    if len(im_pad) != len(freq_ext):
        raise ValueError("Array lens do not match")

    time, fim = FIT(im_pad,freq_ext)
    _sw = np.argsort(time)
    time = time[_sw];# time = time[0:-1:2*extnumber]; time = np.delete(time, [nof//2, nof//2 - 1])
    fim = fim[_sw];# fim = fim[0:-1:2*extnumber]; fim = np.delete(fim, [nof//2, nof//2 - 1])
    return time, fim

def convolution_scaling(freq_c, kernel, pulse, extnumber = 20):
    def edge_function(om, A):
        return A / om

    def adv_edge_function(om, A, B):
        return A / (om + B)

    df = abs(freq_c[0] - freq_c[1])
    nof = freq_c.size
    edge = (2 * extnumber - 1) * (nof // 2)
    if nof % 2 == 1:
        raise ValueError("Odd number of frequencies: unexpected results might happen")
    freq_ext = np.fft.fftfreq(nof * 2 * extnumber) * df * nof * 2 * extnumber
    _sf = np.argsort(freq_ext)
    freq_ext = freq_ext[_sf]

    im = np.multiply(kernel, pulse)
    im_pad = np.pad(im, (edge, edge), 'constant', constant_values=(0, 0))
    adv = True
    if not adv:
        Am = freq_c[0] * im[0]
        Ap = freq_c[-1] * im[-1]

        im_pad[0:edge] = edge_function(freq_ext[0:edge], Am)
        im_pad[-edge:] = edge_function(freq_ext[-edge:], Ap)
    else:
        Am = (freq_c[1] - freq_c[0]) / (1/im[1] - 1/im[0])
        Bm = Am / im[0] - freq_c[0]

        Ap = (freq_c[-2] - freq_c[-1]) / (1/im[-2] - 1/im[-1])
        Bp = Am / im[-1] - freq_c[-1]

        im_pad[0:edge] = adv_edge_function(freq_ext[0:edge], Am, Bm)
        im_pad[-edge:] = adv_edge_function(freq_ext[-edge:], Ap, Bp)


    time, fim = FIT(im_pad, freq_ext)
    _sw = np.argsort(time)
    time = time[_sw];  #time = time[0:-1:2*extnumber]; #time = np.delete(time, [nof//2, nof//2 - 1])
    fim = fim[_sw];  #fim = fim[0:-1:2*extnumber]; #fim = np.delete(fim, [nof//2, nof//2 - 1])
    return time, fim

def rect_pulse(om, gamma):
    T0 = 1/gamma
    return 1 / (1j * om) * (np.exp(1j*om*T0) - 1)

def gauss_pulse(om, gamma):
    """
    Inc. pulse as function of $\omega$ (conj. to time, frequency) (Fourier image)
    T0 - pulse parameter (mean square duration)
    """
    T0 = 1/gamma
    return  np.exp((1j * om * T0 * (1j * om * T0 + 2 * np.pi**2))/ (4*np.pi**2)) * ((2 / np.pi) * T0**2)**(1/4)

def inverse_pulse(om, gamma):
    return 1/(om - 1j*gamma/2)

def gauss_pulse_v2(om, tau):
    #fourier of Exp[-2Ln2*t^2/tau^2]
    return np.exp((-tau**2 * om**2)/(8*np.log(2)))

def delay(freq, shifts, kernel, T0, vg=0.7):
    times = np.empty([len(shifts)], dtype=np.float32)
    fid = np.empty([len(shifts)], dtype=np.float32)
    i = 0

    _, _in =  convolution(freq, np.ones_like(kernel), gauss_pulse(freq, T0), vg)
    for shift in shifts:

        t,f = convolution(freq, kernel, gauss_pulse(freq-shift, T0), vg)
        times[i] = (sum(t*abs(f)**2)/sum(abs(f)**2))


        fid[i] = np.sqrt( sum(abs(f)**2) / sum(abs(_in)**2) )
        i+=1

    return np.asarray(fid), np.asarray(times)


def efficiency(gamma, t, OutPulse):
    SquaredPulse = (abs(OutPulse))**2
    zero = np.argmin(t**2)

    return simps(SquaredPulse[zero:], t[zero:],even='avg') * gamma

def efficiency_sq(gamma, t, SquaredPulse):
    zero = np.argmin(t**2)

    return simps(SquaredPulse[zero:], t[zero:],even='avg') * gamma

def norm_gauss(gamma, t, SquaredPulse):
    return simps(SquaredPulse, t,even='avg') * gamma

def efficiency_map(sc_freq, sc_amp, times, shifts):
    effy = np.empty([len(times), len(shifts)],dtype=np.float32)
    for i in range(len(times)):
        for j in range(len(shifts)):
            time = times[i]
            shift = shifts[j]
            fourier_t, fourier_amp = convolution(sc_freq, sc_amp, inverse_pulse(sc_freq + shift, time), vg=0.7)
            effy[i,j] = efficiency(time, fourier_t, fourier_amp)
    return effy

def unitaryTransform(S):
    U = 1 / np.sqrt(2) * np.asarray([[1., 1.], [1., -1.]])
    return np.matmul(U, np.matmul(S, np.transpose(U)))

class ideal_spectra:
    def __init__(self, omega, gamma, rabi, rabidt, gamma_parasite=0):
        self.om = omega
        self.gm = gamma
        self.rb = rabi
        self.rbdt = rabidt
        self.gmp = gamma_parasite

    def reflection(self):
        return -0.5j * self.gm / (self.om - self.rb**2 / (4*(self.om-self.rbdt)) + 0.5j * (self.gm + self.gmp))

    def transmission(self):
        return 1 + self.reflection()

    def double_entry(self):
        return 1 + 2*self.reflection()