import os
import numpy as np
from .multiwrapper import Parameter, ModelContainer


# TODO: generalize it with the parameter names already present in pyspeckit!
def get_n2dp_model(sp, std_noise, priors=None, npeaks=1, **kwargs):
    # initializing the model parameters
    if priors is None:
        # set up dummy priors for an example run
        # FIXME: the popping techninque, amazing as
        #        it is, is merely an ugly hack!
        #        priors should be initialized from a dict
        priors = [[3, 20], [0, 30], [-30, 30], [0, 1]][::-1] * npeaks

    parlist = []
    for i in range(npeaks):
        tex = Parameter("tex_{}".format(i),
                        r'$\mathrm{{T_{{ex{}}}}}$'.format(i), priors.pop())
        tau = Parameter("tau_{}".format(i), r'$\mathrm{{\tau_{}}}$'.format(i),
                        priors.pop())
        xoff = Parameter("xoff_{}".format(i),
                         r'$\mathrm{{x_{{off{}}}}}$'.format(i), priors.pop())
        sig = Parameter("sig_{}".format(i), r'$\sigma_{}$'.format(i),
                        priors.pop())

        parlist += [tex, tau, xoff, sig]

    n2dp_model = ModelContainer(
        parlist,
        model=sp.specfit.get_full_model,
        std_noise=std_noise,
        xdata=sp.xarr.value,
        ydata=sp.data,
        npeaks=npeaks,
        **kwargs)

    return n2dp_model


def suffix_str(model, snr):
    """ Name id for output files """
    fixed_str = ''.join([{True: 'T', False: 'F'}[i] for i in model.fixed])
    out_suffix = '{}_snr{:n}'.format(fixed_str, snr)
    return out_suffix


def get_pymultinest_dir(output_dir, prefix, suffix, subdir='chains'):
    """ Sets up and returns multinest output directory """
    local_dir = '{}/{}_{}/'.format(subdir, prefix, suffix)
    pymultinest_output = os.path.join(output_dir, local_dir)

    if not os.path.exists(pymultinest_output):
        os.mkdir(pymultinest_output)

    return pymultinest_output
