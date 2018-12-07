"""
Inspired by the gaussian line example from PyMultiNest tutorials:
http://johannesbuchner.github.io/pymultinest-tutorial/example2.html
"""

import os
import numpy as np
import matplotlib.pylab as plt
from astropy import log
try:
    import pymultinest
except ImportError:
    raise ImportError(
        "Abandon the ship! PyMultiNest not found! "
        "You can follow this neat installation walkthrough:\n"
        "http://astrobetter.com/wiki/MultiNest+Installation+Notes")
from .multiwrapper import Parameter, ModelContainer
from .model_nh3 import nh3_spectrum


def get_nh3_model(sp, lines, std_noise, priors=None, npeaks=1, **kwargs):
    """ Set up a new model from ModelContainer """
    if priors is None:
        # set up dummy priors for an example run
        # FIXME: the popping techninque, amazing as
        #        it is, is merely an ugly hack!
        #        priors should be initialized from a dict
        priors = [[5, 25], [2.7315, 10], [10, 17], [0, 1], [0, 20],
                  [0, 1]][::-1] * npeaks

    parlist = []
    for i in range(npeaks):
        tkin = Parameter("tkin_{}".format(i),
                         r'$\mathrm{{T_{{kin{}}}}}$'.format(i), priors.pop())
        tex = Parameter("tex_{}".format(i),
                        r'$\mathrm{{T_{{ex{}}}}}$'.format(i),
                        priors.pop()) # makes no sense setting tex > tkin
        ntot = Parameter("ntot_{}".format(i),
                         r'$\mathrm{{N_{{tot{}}}}}$'.format(i), priors.pop())
        sig = Parameter("sig_{}".format(i), r'$\sigma_{}$'.format(i),
                        priors.pop())
        xoff = Parameter("xoff_{}".format(i),
                         r'$\mathrm{{x_{{off{}}}}}$'.format(i), priors.pop())
        o2p = Parameter("o2p_{}".format(i),
                        r'$\frac{{ortho-NH_3}}{{para-NH_3}}_{}$'.format(i),
                        priors.pop())

        if 'twotwo' not in lines: tkin.freeze(10)
        if 'threethree' not in lines: o2p.freeze(0.5)

        parlist += [tkin, tex, ntot, sig, xoff, o2p]

    nh3_model = ModelContainer(
        parlist,
        model=sp.specfit.get_full_model,
        std_noise=std_noise,
        xdata=sp.xarr.value,
        ydata=sp.data,
        npeaks=npeaks,
        **kwargs)

    return nh3_model


# Setting some plotting defaults here. But why? It's for testing run only.
def get_plot_model_kwargs():
    plot_model_kwargs = dict(
        linestyle='-', c='blue', alpha=0.3, linewidth=0.1, zorder=0.0)
    return plot_model_kwargs


def get_pymultinest_kwargs():
    pymultinest_kwargs = dict(
        sampling_efficiency=0.8,
        #resume = False,
        verbose=True)
    return pymultinest_kwargs


def suffix_str(model, snr):
    """ Name id for output files """
    fixed_str = ''.join([{True: 'T', False: 'F'}[i] for i in model.fixed])
    out_suffix = '{}_snr{:n}'.format(fixed_str, snr)
    return out_suffix


def synthetic_nh3_inference(sp=None, std_noise=None, npeaks=1,
                            lines=['oneone', 'twotwo'], suffix=None,
                            truepars=[10, 5, 14, 0.2, 10.0, 0.5],
                            leastsq_guesses=[10, 5, 14, 0.2, 10.0, 0.5],
                            priors=None, snr=3, snr_line='twotwo',
                            output_dir='', name_id='nh3'):
    """
    A function containing an example script.
    * Generates synthetic ammonia inversion lines
    * Runs PyMultiNest
    * Fits ammonia lines with pyspeckit
    * Plots possible models alongside with minimal-chi^2 fit
    * Plots the global posterior parameter distribution

    Parameters
    ----------
    sp : pyspeckit.spectrum.units.SpectroscopicAxs instance;
         if None, will be automatically generated

    sp_error : variance of the channel noise;
               ignored if `sp` is None;
    """
    # generating fake ammonia inversion lines
    if sp is None:
        xvals = np.linspace(-20, 40, 1000)
        sp = nh3_spectrum(truepars, xvals, seed=None, lines=lines,
            snr=snr, snr_line='twotwo')
        std_noise = sp.error.mean()

    nh3_model = get_nh3_model(
        sp, lines, std_noise, priors=priors, npeaks=npeaks)
    fig_dir = os.path.join(output_dir, 'figs')
    if suffix is None:
        suffix = suffix_str(nh3_model, snr)
    chain_dir = os.path.join(output_dir, 'chains')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(chain_dir):
        os.mkdir(chain_dir)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    chain_file = os.path.join(chain_dir, '{}-'.format(npeaks))

    # fire up multinest
    pymultinest.run(nh3_model.log_likelihood, nh3_model.prior_uniform,
        nh3_model.npars, outputfiles_basename=chain_file,
        **get_pymultinest_kwargs())

    # and parse the results as sensible output
    a = pymultinest.Analyzer(
        outputfiles_basename=chain_file, n_params=nh3_model.npars)

    # plot the distribution of a posteriori possible models
    fig_models = plt.figure()
    sp.plotter(figure=fig_models, lw=1.0)
    sp.specfit(fittype='cold_ammonia', guesses=leastsq_guesses,
               fixed=list(nh3_model.fixed))
    for pars in a.get_equal_weighted_posterior()[::10, :-1]:
        tmp = plt.plot(sp.xarr.value, nh3_model.model(pars=pars),
                       **get_plot_model_kwargs())
    plt.savefig(os.path.join(fig_dir,
                "{}nh3-fit-x{}.pdf".format(suffix, npeaks)))
    plt.show()

    # There used to be a plotting end to this function. Need to
    # replace it with something more informative...
    # TODO: check 'mean' & 'sigma' values for those nodes
    #       under a.get_mode_stats()! looks like something
    #       local that I can compare to pyspeckit...
    # TODO: also, looks like there are already pre-computed
    #       stats in a.get_stats()! look under
    #       a.get_stats()['marginals'][npar] for more info

    a_lnZ = a.get_stats()['global evidence']
    log.info('log Z for model with 1 line = %.1f' % (a_lnZ / np.log(10)))

    return a


def main():
    """ Runs a sample script and outputs in the new directory """

    output_dir = 'bayesianity/'  # The trailing slash is mandatory!
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    synthetic_nh3_inference(snr=3, output_dir=output_dir)


if __name__ == "__main__":
    main()
