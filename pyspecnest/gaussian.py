""" A gaussian model wrapper """
from .multiwrapper import Parameter, ModelContainer


def get_gauss_model(sp, std_noise, priors=None, npeaks=1, **kwargs):
    # initializing the model parameters
    if priors is None:
        # set up dummy priors for an example run
        # FIXME: the popping techninque, amazing as
        #        it is, is merely an ugly hack!
        #        priors should be initialized from a dict
        # TODO: can set prors on the velocity to be between
        #       xarr.min and xarr.max...
        priors = [[0, 10], [42, 48], [0, 1]][::-1] * npeaks

    parlist = []
    for i in range(npeaks):
        amp = Parameter("amp_{}".format(i), r'$\mathrm{{A_{}}}$'.format(i),
                        priors.pop())
        xoff = Parameter("xoff_{}".format(i),
                         r'$\mathrm{{x_{{off{}}}}}$'.format(i), priors.pop())
        sig = Parameter("sig_{}".format(i), r'$\sigma_{}$'.format(i),
                        priors.pop())

        parlist += [amp, xoff, sig]

    gauss_model = ModelContainer(
        parlist,
        model=sp.specfit.get_full_model,
        std_noise=std_noise,
        xdata=sp.xarr.value,
        ydata=sp.data,
        npeaks=npeaks,
        **kwargs)

    return gauss_model
