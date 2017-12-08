from .blackbox import toy_observation
from ..gaussian import get_gauss_model
import numpy as np

def make_test_spec_model(npeaks=2):
    sp = toy_observation(snr=15, seed=0)
    sp.error[:] = sp.header['NOISERMS']
    sp.specfit.fitter = sp.specfit.Registry.multifitters['gaussian']
    sp.specfit.fitter.npeaks = npeaks

    priors = [[0, 10], [30, 55], [0, 5]][::-1] * npeaks

    spec_model = get_gauss_model(sp, sp.error, priors=priors, npeaks=npeaks)

    return spec_model


def test_basic(npeaks=2):
    """ Sanity check """
    spec_model = make_test_spec_model(npeaks=2)

    log_like = spec_model.log_likelihood([3, 45, 0.2] * npeaks,
                                         spec_model.npars, spec_model.dof)
    assert np.isfinite(log_like)


def test_xoff_transform():
    """
    Regression test for #1.

    Make sure log-likelihood is coordinate transformation invariant.
    """
    spec_model = make_test_spec_model(npeaks=2)

    log_L_standard = spec_model.log_likelihood([1, 40, 2, 1, 42, 2], 6, 6)
    log_L_transformed = spec_model.xoff_symmetric_log_likelihood(
                                               [1, 41, 2, 1, 2, 2], 6, 6)

    assert log_L_standard == log_L_transformed
