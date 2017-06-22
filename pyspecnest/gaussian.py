""" A gaussian model wrapper """
from .multiwrapper import Parameter, ModelContainer


def get_gauss_model(sp, std_noise, priors=None, npeaks=1):
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
        amp = Parameter("amp_%i" % i, r'$\mathrm{A}$', priors.pop())
        xoff = Parameter("xoff_%i" % i, r'$\mathrm{x_{off}}$', priors.pop())
        sig = Parameter("sig_%i" % i, r'$\sigma$', priors.pop())

        parlist += [amp, xoff, sig]

    gauss_model = ModelContainer(
        parlist,
        model=sp.specfit.get_full_model,
        std_noise=std_noise,
        xdata=sp.xarr.value,
        ydata=sp.data)

    return gauss_model
