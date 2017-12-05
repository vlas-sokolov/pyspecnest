import numpy as np
import astropy.units as u
from astropy import log
from pyspeckit.spectrum.units import SpectroscopicAxis
from pyspeckit.spectrum.models.inherited_gaussfitter import gaussian
from pyspeckit.spectrum.classes import Spectrum

def toy_observation(snr = 2, debug = False, seed = 0,
                    v1=10, v2=70, nchan=1000,
                    truth_narrow=[4, 40, 1.0],
                    truth_wide=[2, 41, 3.0]):
    np.random.seed(seed)
    if debug: # <CheatMode>
        log.setLevel('DEBUG')

    # initialize the spectral axis
    xunit, bunit = 'km/s', 'K'
    refX = 120 * u.GHz
    log.debug("Genarating a spactral axis instance from {}"
              " to {} {}".format(v1, v2, xunit))
    xarr = SpectroscopicAxis(np.linspace(v1, v2, nchan) * u.Unit(xunit),
                             refX = refX, velocity_convention = 'radio')

    # generate a spectrum approximated by a gaussian
    log.debug("Gaussian parameters for the"
              " narrow component: {}".format(truth_narrow))
    log.debug("Gaussian parameters for the"
              " wide component: {}".format(truth_wide))
    true_data_narrow = gaussian(xarr, *truth_narrow)
    true_data_wide = gaussian(xarr, *truth_wide)

    true_total = true_data_narrow + true_data_wide
    signal_peak = true_total.max()
    log.debug("For a signal-to-noise ratio of {} the square root of noise"
              " variance is {:.2f} {}.".format(snr, signal_peak / snr, bunit))
    noise = np.random.normal(loc = 0, scale = signal_peak / snr,
                             size = xarr.size)

    observed = true_total + noise

    log.setLevel('INFO') # <\CheatMode>

    # make a spectrum class instance in Tmb units
    xarr._make_header()

    sp = Spectrum(xarr = xarr, data = observed,
                  unit = u.Unit(bunit), header = {})
    sp.header['NPEAKS'] = 2
    sp.header['NOISERMS'] = round(signal_peak / snr, 4)
    for comp, name in zip([truth_narrow, truth_wide], ['1', '2']):
        sp.header['AMP_'  + name] = comp[0]
        sp.header['XOFF_' + name] = comp[1]
        sp.header['SIG_'  + name] = comp[2]
    return sp

def main():
    import matplotlib.pyplot as plt
    sp = toy_observation(snr = 10)
    sp.plotter()
    plt.show()

if __name__ == '__main__':
    main()
