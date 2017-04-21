import numpy as np
import astropy.units as u
import pyspeckit
from pyspeckit.spectrum.models import ammonia_constants, ammonia
from pyspeckit.spectrum.units import SpectroscopicAxes


def nh3_spectrum(pars, kms_vals, lines=['oneone', 'twotwo'], snr=5,
                 snr_line='oneone', seed=None):
    """
    Write me up!
    """
    # building a spectroscopic axis
    xarr_dict = {}
    for line in lines:
        xarr_dict[line] = pyspeckit.units.SpectroscopicAxis(kms_vals, 'km/s')
        xarr_dict[line].refX = ammonia_constants.freq_dict[line] * u.Hz
        xarr_dict[line].velocity_convention = 'radio'

        # TODO: bug... see if #182 fixes it
        #xarr_dict[line].convert_to_unit('GHz')

        xarr_dict[line] = xarr_dict[line].as_unit('GHz')

    xarr = SpectroscopicAxes([xarr_dict[line] for line in lines])
    xarr.refX = ammonia_constants.freq_dict['oneone'] * u.Hz
    xarr.velocity_convention = 'radio'
    xarr.convert_to_unit('km/s')

    # making synthetic spectra
    sp = pyspeckit.Spectrum(data=np.zeros_like(xarr.value), xarr=xarr)
    sp.Registry.add_fitter('cold_ammonia', ammonia.cold_ammonia_model(), 6)
    sp.specfit.fitter = sp.specfit.Registry.multifitters['cold_ammonia']
    sp.specfit.fittype = 'cold_ammonia'
    signal = sp.specfit.get_full_model(pars=pars)
    sp.data = signal
    line_mask = xarr_dict[snr_line].min(), xarr_dict[snr_line].max()  # GHz
    std_noise = sp.slice(*line_mask).data.max() / snr
    np.random.seed(seed=0)
    try:
        noise = np.random.normal(0, std_noise, sp.xarr.size)
    except ValueError:  # ValueError("scale <= 0")
        noise = 0
    sp.data += noise
    sp.error = np.zeros_like(sp.data) + std_noise

    return sp
