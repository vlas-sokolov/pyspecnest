"""
Inspired by the gaussian line example from PyMultiNest tutorials:
http://johannesbuchner.github.io/pymultinest-tutorial/example2.html

I'm going to expand this model to include
* easy-to-specify priors,
* for pyspeckit spectral models,
* of flexible npeak values;
* and eventually compare Bayes factors for variable dof's
"""
import numpy as np
from astropy import log
import scipy.stats
try:
    import pymultinest
except ImportError:
    raise ImportError(
        "Abandon the ship! PyMultiNest not found! "
        "You can follow this near installation break-down:\n"
        "http://astrobetter.com/wiki/MultiNest+Installation+Notes")
import matplotlib.pylab as plt
import corner
from collections import OrderedDict

nchan = 400
std_noise = 1
real_model_pars = [0.5, 0.013, 0.2]
corner_quartiles = [0.16, 0.5, 0.84]  # +-1\sigma

# defining the spectral axis:
x = np.linspace(0, 1, nchan)


# defining the spectral model for multinest
def model(pos1, width, height1, height2=0):
    """
    A model function for 2 gaussians.
    Same width, fixed offset.
    """
    pos2 = pos1 + 0.05
    return height1 * scipy.stats.norm.pdf(x, pos1, width) + \
        height2 * scipy.stats.norm.pdf(x, pos2, width)


# the data we're going to model with multinest
noise = np.random.normal(0, std_noise, nchan)
ydata = model(*real_model_pars) + noise


class Parameter:
    """
    This thingy here is ought to contain everything
    related to individual parameters - priors, names, and
    convenience methods for getting prior multipliers
    used later as pymultinest feed.
    """

    def __init__(self, name, nametex=None, prior=[0, 1], **kwargs):
        self.name = name
        self.prior = prior
        if nametex:
            self.nametex = nametex

    def __repr__(self):
        return ("Parameter instance \"{}\", defined from {} to {}".format(
            self.name, self.prior[0], self.prior[1]))

    def get_latex(self):
        try:
            return self.nametex
        except AttributeError:
            return self.name

    def get_uniform_mods(self):
        return self.prior[1] - self.prior[0], self.prior[0]


class ModelContainer(OrderedDict):
    """
    A collection of model methods and attributes.
    """

    def __init__(self, parlist, **kwargs):
        super(ModelContainer, self).__init__()

        self.npars = len(parlist)
        self.parlist = parlist
        for par in parlist:
            # let it all pass, pandas style
            setattr(self, par.name, par)
            self[par.name] = par

        self.update_model(**kwargs)
        self.update_data(**kwargs)

    def update_model(self, model=None, **kwargs):
        if model is not None: self.model = model

    def update_data(self, xdata=None, ydata=None, std_noise=None, **kwargs):
        if xdata is not None: self.xdata = xdata
        if ydata is not None: self.ydata = ydata
        if std_noise is not None: self.std_noise = std_noise

    def get_latex_names(self):
        return [self[par].get_latex() for par in self]

    def prior_uniform(self, cube, ndim, nparams):
        """
        If this set to simply pass, defaults to uniform prior from 0 to 1.

        The log-uniform prior below seems to speed up the sampling, but
        that's it - the end posterior shape seems to remain the same.
        """
        #TODO: test how sensitive the posterior is to different priors

        #NOTE: sometimes priors can cause segfaults.
        #      I have a bad feeling about this . . .

        for i, par in enumerate(self):
            a, b = self[par].get_uniform_mods()
            cube[i] = cube[i] * a + b

    def log_likelihood(self, cube, ndims, nparams):
        """
        Quick-grok info:
        * Uses args to get the parameter vector
        * Uses global variables to get f(pars) - data
        * Calculates the "astronomer's chi-square"
        * Returns -0.5 x chi^2
        """
        #TODO: hack this to get variable npeaks
        #TODO: wrap this around pyspeckit n_modelfunc for fun and profit

        par_list = [cube[i] for i in range(ndims)]
        ymodel = self.model(*par_list)
        log_L = (-0.5 * ((ymodel - self.ydata) / self.std_noise)**2).sum()
        return log_L


pos = Parameter("pos_0", r'$pos_0$', [0.4, 0.6])
width = Parameter("width_0", r'$width_0$', [0.001, 0.1])
height = Parameter("height_0", r'$height_0$', [0.0, 0.5])

gauss_model = ModelContainer(
    [pos, width, height],
    model=model,
    xdata=x,
    ydata=ydata,
    std_noise=std_noise)

# because we want to `cat *.eggs > /dev/basket`
plot_model_kwargs = dict(linestyle='-', c='blue', alpha=0.3, lw=0.1)
plot_data_kwargs = dict(drawstyle='steps', linestyle='-', c='k', label='data')
corner_kwargs = dict(
    labels=gauss_model.get_latex_names(),
    show_titles=True,
    title_fmt='.3f',
    smooth=0.5,
    bins=40,
    #plot_contours = False, plot_density = False,
    quantiles=corner_quartiles,
    truths=real_model_pars)
pymultinest_kwargs = dict(sampling_efficiency=0.8, resume=False, verbose=True)
pymultinest_output = 'out/1-'

# fire up the multinest blackbox
pymultinest.run(gauss_model.log_likelihood, gauss_model.prior_uniform,
                gauss_model.npars, outputfiles_basename=pymultinest_output,
                **pymultinest_kwargs)

# and parse the results as sensible output
a = pymultinest.Analyzer(outputfiles_basename=pymultinest_output,
                         n_params=gauss_model.npars)

# plot the distribution of posteriori possible models
fig_models = plt.figure()
plt.plot(x, ydata, **plot_data_kwargs)
for (pos1, width, height1) in a.get_equal_weighted_posterior()[::10, :-1]:
    tmp = plt.plot(x, model(pos1, width, height1, 0), **plot_model_kwargs)
plt.show()

a_lnZ = a.get_stats()['global evidence']
log.info('log Z for model with 1 line = %.1f' % (a_lnZ / np.log(10)))

fig_corner, axarr = plt.subplots(gauss_model.npars, gauss_model.npars)
corner.corner(a.data[:, 2:], fig=fig_corner, **corner_kwargs)
plt.show()
