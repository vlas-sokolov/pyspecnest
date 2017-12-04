from __future__ import division
import numpy as np
from astropy import log
from itertools import compress
from collections import OrderedDict


class Parameter:
    """
    This class framework is ought to contain everything
    related to individual parameters - priors, names, and
    convenience methods for getting prior multipliers
    used later as pymultinest feed.
    """

    def __init__(self, name, nametex=None, prior=[0, 1], fixed=False,
                 **kwargs):
        self.name = name
        self.prior = prior
        if nametex:
            self.nametex = nametex
        self.fixed = fixed
        if fixed:
            self.freeze(**kwargs)

    def __repr__(self):
        return ("Parameter instance \"{}\", defined from {} to {}".format(
            self.name, self.prior[0], self.prior[1]))

    def freeze(self, value=None, **kwargs):
        """
        Fix the parameter so it won't get in a way of the
        multidimensional analysis about to unfold here.
        """
        self.fixed = True
        value = value or np.mean(self.prior)
        if not (self.prior[0] <= value <= self.prior[1]):
            log.warn("Fixing parameter {} outside"
                     " of its prior".format(self.name))
        self.fixed_value = value

        self.frozen_prior = self.prior[:]
        self.prior = [value, value]

    def unfreeze(self):
        # NOTE: this probably isn't a good approach...
        #       a better one would be to make a prior class
        #       with bounds, PDF, *and* a "frozen" value
        self.fixed = False
        if hasattr(self, 'fixed_value'):
            delattr(self, 'fixed_value')
        if hasattr(self, 'frozen_prior'):
            self.prior = self.frozen_prior[:]
            delattr(self, 'frozen_prior')

    def get_name(self, latex=False):
        """
        Returns parameter name.

        Parameters
        ----------
        latex : bool; return a \LaTeX name, in available
        """
        try:
            name = self.name if not latex else self.nametex
        except AttributeError:
            name = self.name

        return name

    def get_uniform_mods(self):
        return self.prior[1] - self.prior[0], self.prior[0]

    def limited_lower_mods(self, mod):
        return self.prior[1] - mod, mod

    def limited_upper_mods(self, mod):
        return mod - self.prior[0], self.prior[0]


def get_xoff_transform(npeaks):
    """
    Returns a transformation matrix between two sets of line centroid
    coordinates.

    Transforms (x1, x2, x3, ..., xn) into (xmean, dx1, dx2, ... dxn-1).
    """
    T = np.zeros(shape=(npeaks, npeaks))
    # matrix row for mean line centroid calculation
    T[0, :] = 1 / npeaks
    # fill the rows for the centroid intervals
    for i in range(1, npeaks):
        T[i, i - 1] = -1
        T[i, i] = 1

    return T


def xoff_transform(f_loglike):
    """
    Hijack a `cube`-based method and inject a custom set of transformed
    parameters into it. The coordinate transform is pre-computed by the
    the inverse matrix from the `get_xoff_transform` method.
    """
    def log_like_transformed(*args, **kwargs):
        # I know I know this can be write in a simpler way, but we want
        # to really avoid making new arrays in memory and I am not sure
        # what type the `cube` variable is inside PyMultinest
        cube = args[1]
        trans_xoff_vector = [cube[i] for i, is_xoff in
                             enumerate(kwargs["xoff_pars"]) if is_xoff]

        xoff_vector = np.dot(kwargs["inv_xoff_T"], trans_xoff_vector)

        # inject xoffs into cube
        xoff_idx = 0
        for i, is_xoff in enumerate(kwargs["xoff_pars"]):
            if is_xoff:
                cube[i] = xoff_vector[xoff_idx]
                xoff_idx += 1

        # compute the log-likelihood in the (x1, x2, x3, ..., xn) system
        f_loglike(*args)

    return log_like_transformed


class ModelContainer(OrderedDict):
    """
    A collection of model methods and attributes.
    """

    def __init__(self, parlist, **kwargs):
        super(ModelContainer, self).__init__()

        for par in parlist:
            # let it all pass, pandas style
            setattr(self, par.name, par)
            self[par.name] = par

        self.std_noise = kwargs.pop('std_noise', None)
        self.npeaks = kwargs.pop('npeaks', 1)
        self.xoff_key = kwargs.pop('xoff_key', 'xoff_')

        self.update_model(**kwargs)
        self.update_data(**kwargs)

        # generate the coordinate transformation matrices
        T = get_xoff_transform(self.npeaks)
        # (x1, x2, x2, ..., xn) --> (xmean, dx1, dx2, ... dxn-1)
        self.xoff_T = T
        # (xmean, dx1, dx2, ... dxn-1) --> (x1, x2, x2, ..., xn)
        self.inv_xoff_T = np.linalg.inv(T)

    # TODO: setting a __repr___ would be nice...

    @property
    def npars(self):
        """ Total number of model parameters """
        return len(self)

    @property
    def dof(self):
        """
        Get degrees of freedom for the model.
        Note that this is the dof count for the
        model only, not for the chi^2 or log(L).
        """
        dof = (~self.fixed).sum()
        return dof

    @property
    def fixed(self):
        """ Returns list of booleans indicating whether pars are frozen """
        return np.array([self[par].fixed for par in self])

    def update_model(self, model=None, **kwargs):
        if model is not None: self.model = model

    def update_data(self, xdata=None, ydata=None, std_noise=None, **kwargs):
        if xdata is not None: self.xdata = xdata
        if ydata is not None: self.ydata = ydata
        if std_noise is not None: self.std_noise = std_noise

    def nonfixed(self, attr_list, slack=False):
        """
        Given an iterable of (npars,) shape, reduces it to
        (dof,) shape by throwing out indices of frozen variables
        """
        if slack:
            return attr_list

        return list(compress(attr_list, ~self.fixed))

    def get_nonfixed_slice(self, shape, axis=0):
        """ Returns a non-frozen slice for given shape and axis """
        nf_slice = [
            slice(None, None, None) if i != axis else ~self.fixed
            for i in range(len(shape))
        ]
        return np.s_[nf_slice]

    def get_names(self, latex=False, no_fixed=False):
        """
        Returns a list of model parameter names.

        Parameters
        ----------
        latex : bool; return LaTeX names if available, defaults to False

        no_fixed : bool; omit frozen parameters, defaults to False
        """
        return self.nonfixed(
            [self[par].get_name(latex) for par in self], slack=not no_fixed)

    def prior_uniform(self, cube, ndim, nparams):
        """ Pass ModelContainer uniform priors to MultiNest """
        for i, par in enumerate(self):
            a, b = self[par].get_uniform_mods()
            cube[i] = cube[i] * a + b

    def prior_restricted_xoff(self, cube, ndim, nparams):
        """
        Impose xoff_i < xoff_j for all i > j;
        Flat priors otherwise. Doesn't quite work yet....
        """
        # FIXME: doesn't converge where we want it, needs debugging.
        for i, par in enumerate(self):
            if (i == 10) or (i == 16):
                # FIXME: this has to be generalised...
                a, b = self[par].limited_lower_mods(cube[i - 6])
            else:
                a, b = self[par].get_uniform_mods()
            cube[i] = cube[i] * a + b

    def log_likelihood(self, cube, ndims, nparams):
        """ Returns -0.5 x chi^2 """
        par_list = [cube[i] for i in range(ndims)]
        ymodel = self.model(pars=par_list)
        log_L = (-0.5 * ((ymodel - self.ydata) / self.std_noise)**2).sum()
        return log_L

    @property
    def xoff_pars(self, key=None):
        """
        Returns a boolean array signaling if a parameter in `cube` is in xoff

        Needed for conversion between parameter name space and the free-styled
        indexing in the input format of multinest.
        """
        if key is None:
            key = self.xoff_key

        return [key in parname for parname in self]

    @xoff_transform
    def xoff_symmetric_log_likelihood(self, cube, ndims, nparams):
        return self.log_likelihood(self, cube, ndims, nparams)

    #@xoff_transform
    #def xoff_symmetric_prior_uniform(self, cube, ndim, nparams):
    #    pass
