""" Functions that read in and analyze MultiNest chains """

from __future__ import division
import os
import functools
from itertools import combinations
import numpy as np
from astropy.io import fits
from astropy import log
from uncertainties import ufloat, unumpy
import pymultinest

# TODO: goal #2 is writing output fits files with MAP/MLE parameters
# TODO: write up another function that makes npeaks map, but this time
#       by imposing strict Bayes factor cuts.

def _tinker_header(h, ctype3='', bunit='', flatten=False):
    """
    Convert a 3d header to accommodate utility-storing data.
    """
    # FIXME this breaks if header is None!
    # adapt dummy header to the task at hand
    h['CTYPE3'] = ctype3
    h['BUNIT'] = bunit
    h['CDELT3'] = 1
    h['CRVAL3'] = 1
    h['CRPIX3'] = 1

    if flatten:
        h['NAXIS'] = 2

    for key in h.keys():
        if key.startswith('PLANE'):
            h.pop(key, None)
        if flatten and key.endswith('3'):
            h.pop(key, None)

    return h


# If MultiNest truncates the path+filenames beyond this, the files are lost!
_multikeys_unique = {'e': 'ev.dat',
                     'IS.i': 'IS.iterinfo',
                     'IS.po':'IS.points',
                     'IS.pt':'IS.ptprob',
                     'l': 'live.points',
                     'ph': 'phys_live.points',
                     'po': 'post_equal_weights.dat',
                     'r': 'resume.dat',
                     'su': 'summary.txt',
                     'st': 'stats.dat',
                     '.': '.txt'}

_multikeys_shortest = {'ev.dat':'e',
                       'IS.iterinfo':'IS.i',
                       'IS.points':'IS.po',
                       'IS.ptprob':'IS.pt',
                       'live.points':'l',
                       'phys_live.points':'ph',
                       'post_equal_weights.dat':'po',
                       'resume.dat':'r',
                       'summary.txt':'su',
                       'stats.dat':'st',
                       '.txt':'.'}


def fix_multimess(path):
    """
    MultiNest truncates full filenames to data files to a 100 characters.

    https://ccpforge.cse.rl.ac.uk/gf/project/multinest/tracker/ \
    ?action=TrackerItemEdit&tracker_item_id=1732
    """
    flst = os.listdir(path)

    for f in flst:
        for fshort, flong in _multikeys_unique.items():
            if f[2:].startswith(fshort) and flong not in f:
                srcfile = os.path.join(path, f)
                dstfile = os.path.join(path, f[:2]+flong)
                os.rename(srcfile, dstfile)


def fix_truncated(analyzer_initter):
    """
    Due to a suspected case of Jurassic programming the MultiNest file
    names (with path!) are truncated to 100 characters.

    Line 28 in posterior.F90 reads: character(LEN=100)root

    So, the best clutch is to check if truncated files can be read still as
    the Analyzer instance is being initialized and to modify its file names
    in place.
    """
    @functools.wraps(analyzer_initter) # preserves __name__, docstring
    def analyzer_filenames_restored(*args, **kwargs):
        a = analyzer_initter(*args, **kwargs)

        for file_attr, key in zip(
                ['stats_file', 'data_file', 'equal_weighted_file'],
                ['stats.dat', '.txt', 'post_equal_weights.dat']):
            fname = a.__getattribute__(file_attr)
            if not os.path.exists(fname):
                dirname = os.path.dirname(fname)
                prefix = os.path.basename(fname).replace(key, '')
                try:
                    f_lst = os.listdir(os.path.dirname(fname))
                except (FileNotFoundError, OSError) as e:
                    return a # should be handled upstream
                truncated_paths = [os.path.join(dirname, f) for f in f_lst
                        if f.startswith(prefix+_multikeys_shortest[key])]
                if len(truncated_paths) > 1:
                    raise log.warning("Multiple candidates for file truncated"
                                      " by MultiNest. Bad things may happen!")
                elif not len(truncated_paths):
                    return a
                a.__setattr__(file_attr, truncated_paths[0])

        return a

    return analyzer_filenames_restored


def _get_analyzer(output_dir, name_id, suffix, ndims, chain_prefix='1-'):
    chains_dir = '{}chains/{}_{}/{}'.format(output_dir, name_id, suffix,
                                            chain_prefix)
    a = pymultinest.Analyzer(outputfiles_basename=chains_dir, n_params=ndims)
    return a

@fix_truncated
def get_analyzer(*args, **kwargs):
    return _get_analyzer(*args, **kwargs)


def analyzer_xy(x, y, npeaks, output_dir, name_id='g35-nh3', npars=6):
    """ Initializes Analizer instances; prone to truncated filenames """
    suffix = 'x{}y{}'.format(x, y)
    return get_analyzer(
        output_dir=output_dir,
        name_id=name_id,
        suffix=suffix,
        ndims=npars * npeaks,
        chain_prefix='{}-'.format(npeaks))


def lnK_xy(peaks=[1, 2, 3], silent=False, dict_key_formatter='{}/{}',
           **kwargs):
    """
    Returns Bayes factors between given peak values.

    Additional keyword arguments are passed to analyzer_xy
    """
    # make a list of pymultinest.Analyzer instances
    x, y = kwargs.pop('x', None), kwargs.pop('y', None)
    peak_stats = [analyzer_xy(x, y, p, **kwargs) for p in peaks]

    # raise warnings for non-existing files if not silent
    for a in peak_stats:
        if not os.path.exists(a.data_file):
            if not silent:
                log.warn("file {} does not exist, expect NaN values"
                         " in the output".format(a.data_file))

    # compute the Bayes factors for all combinations of npeaks
    ln_K_dict = {}
    for i, j in combinations(peaks, 2):
        try:
            ln_K = bayes_factor(peak_stats[j - 1], peak_stats[i - 1])
        except IOError:
            ln_K = ufloat(np.nan, np.nan)
        if not silent:
            log.info("Odds of a {}-component model over {}-component"
                     " one: ln(K) = {}".format(j, i, ln_K))
        ln_K_dict[dict_key_formatter.format(j, i)] = ln_K

    return ln_K_dict


def lnZ_xy(peaks=[1, 2, 3], silent=False, lnZ0=None, **kwargs):
    """
    Returns model evidence for given npeak values.

    Additional keyword arguments are passed to analyzer_xy
    """
    # peaks = 0 requires special treatment
    try:
        peaks = peaks[:]
        peaks.pop(peaks.index(0))
        do_0 = True
    except ValueError:
        do_0 = False

    # make a list of pymultinest.Analyzer instances
    x, y = kwargs.pop('x', None), kwargs.pop('y', None)
    peak_stats = [analyzer_xy(x, y, p, **kwargs) for p in peaks]

    # raise warnings for non-existing files if not silent
    for a in peak_stats:
        if not os.path.exists(a.data_file):
            if not silent:
                log.warn("file {} does not exist, expect NaN values"
                         " in the output".format(a.data_file))

    # compute the Bayes factors for all combinations of npeaks
    ln_Z_npeaks = [ln_Z(a) for a in peak_stats]

    if do_0:
        if lnZ0 is None:
            raise ValueError("lnZ0 must be supplied for peaks = 0")
        ln_Z_npeaks = [ufloat(*lnZ0)] + ln_Z_npeaks
    return ln_Z_npeaks


def get_global_evidence(a):
    """
    PyMultiNest's Analyzer has a get_stats() method, but it's
    a bit too sluggish if all we want is to get the global
    evidence out. This is a hack around the issue.
    """
    stats_file = open(a.stats_file)
    lines = stats_file.readlines()
    stats = {}
    a._read_error_into_dict(lines[1], stats)
    Z_str = 'Nested Importance Sampling Global Log-Evidence'
    Z = stats[Z_str.lower()]
    Zerr = stats[(Z_str + ' error').lower()]

    stats['global evidence'] = Z
    stats['global evidence error'] = Zerr

    return stats


def get_stats_fast(a, stat=''):
    """
    PyMultiNest's `get_best_fit` method reads the whole posterior in the
    memory, which could be very slow if we have to read ~10k posteriors in
    one go just to get the best fit statistics - easier to parse the stats
    file already generated with MultiNest.

    Parameters
    ----------
    a : PyMultiNest Analyzer class instance

    stat : str, one of the ['mean', 'mle', 'map'] - a statistic to read out

    Returns
    -------
    stat : ndims-shaped array for MLE or MAP, ndims x 2 shape for mean
           statistics (for mean and sigma values)
    """
    trigger = {'mean': 'Mean',
               'mle': 'Maximum Likelihood',
               'map': 'MAP Parameters'}

    # Find parameter value
    sweet_spot = False
    stat_result = []

    stats_file = open(a.stats_file)
    for l in stats_file.readlines():
        if trigger[stat] in l:
            sweet_spot = True
            continue
        if sweet_spot:
            if l.startswith('Dim No.'):
                continue
            line_arr = np.fromstring(l.strip(), sep=' ')
            if line_arr.size == 0:
                sweet_spot = False
            else:
                stat_result.append(line_arr)

    # Find parameter uncertainty
    sweet_spot = False
    sigma_result = []

    stats_file = open(a.stats_file)
    for l in stats_file.readlines():
        if 'Sigma' in l:
            sweet_spot = True
            continue
        if sweet_spot:
            if l.startswith('Dim No.'):
                continue
            line_arr = np.fromstring(l.strip(), sep=' ')
            if line_arr.size == 0:
                sweet_spot = False
            else:
                sigma_result.append(line_arr)

    return np.squeeze(np.array(stat_result)[:, 1:].T), np.squeeze(np.array(sigma_result)[:, 2:].T)


def get_stats(a, mode="slow"):
    if mode == "slow":
        return a.get_stats()
    if mode == "fast":
        return get_global_evidence(a)
    else:
        raise ValueError("mode should be one of the ['slow', 'fast']")


def ln_Z(a):
    try:
        stats = get_stats(a, 'fast')
        ln_z = ufloat(stats['global evidence'], stats['global evidence error'])
    except IOError:
        ln_z = ufloat(np.nan, np.nan)

    return ln_z


def bayes_factor(a1, a2):
    """ Returns ln(K) for two Analyzer instances. """

    ln_K = ln_Z(a1) - ln_Z(a2)

    return ln_K


def cube_K(shape, rms, data, peaks=[0, 1, 2, 3], origin=(0, 0),
           header=None, writeto=None, **kwargs):
    """
    Construct a fits HDU with ln(K) values for all xy positions in a
    cube of a given shape. Optionally, writes a fits file.

    Additional keyword args are passed to lnK_xy function.
    """

    if origin != (0, 0):
        # TODO: implement this
        raise NotImplementedError("wip")

    Zs = cube_Z(shape, rms, data, peaks=peaks, origin=origin,
                header=header, writeto=None, **kwargs).data

    # this -2 denotes the that the K array is a difference of Z array
    # layers. It's 2, and not 1, because there are error layers as well
    zsize_K = Zs.shape[0] // 2 - 1
    lnKs = np.empty(shape=(zsize_K, ) + shape)
    lnKs.fill(np.nan)
    err_lnKs = lnKs.copy()
    for i in np.arange(zsize_K) + 1:
        # for all (i, j) such that i = j + 1
        Z_i = unumpy.uarray(Zs[i], Zs[i + zsize_K + 1])
        Z_j = unumpy.uarray(Zs[i - 1], Zs[i + zsize_K])
        K_ij = Z_i - Z_j

        lnKs[i - 1] = unumpy.nominal_values(K_ij)
        err_lnKs[i - 1] = unumpy.std_devs(K_ij)

    header = _tinker_header(header, ctype3='BAYES FACTORS', bunit='ln(Zi/Zj)')

    hdu = fits.PrimaryHDU(np.vstack([lnKs, err_lnKs]), header)
    for i in np.arange(zsize_K) + 1:
        head_key = 'lnK({}/{})'.format(i, i - 1)
        hdu.header['PLANE{}'.format(i)] = head_key
    for i in np.arange(zsize_K) + zsize_K + 1:
        head_key = 'err lnK({}/{})'.format(i, i - 1)
        hdu.header['PLANE{}'.format(i)] = head_key

    if writeto:
        hdu.writeto(writeto, overwrite=True)

    return hdu


def get_zero_evidence(data, rms, normalize=True):
    if normalize and len(data.shape) < 3:
        norm_C = -data.shape[0] * np.log(np.sqrt(2 * np.pi) * rms)
    elif not normalize:
        norm_C = 0
    else:
        raise NotImplementedError

    ln_Z0 = norm_C - (np.square(data) / (2 * rms**2)).sum(axis=0)
    return ln_Z0


def cube_Z(shape, rms, data, peaks=[0, 1, 2, 3], origin=(0, 0), header=None,
           writeto=None, normalize=False, ln_Z0_arr=None, **kwargs):
    """
    Construct a fits HDU with ln(K) values for all xy positions in a
    cube of a given shape. Optionally, writes a fits file.

    Additional keyword args are passed to lnK_xy function.
    """

    if origin != (0, 0):
        # TODO: implement this
        raise NotImplementedError("wip")

    lnZs = np.empty(shape=(len(peaks), ) + shape)
    lnZs.fill(np.nan)
    err_lnZs = lnZs.copy()

    if 0 in peaks and ln_Z0_arr is None:
        # NOTE: no normalization would in this case would imply
        #       that not a full likelihood function passed to
        #       the MultiNest was of the `chisq / 2` form and not
        #       a fully normalized ln(L) function.
        #
        #       i.e., if the median of evidence distributions
        #       differs by thousands, it is a strong hint that
        #       normalization differs for zero and non-zero models...
        # NOTE: this only works for the uniform noise case!
        #       if you're in a situation where you've constructed your
        #       joint likelihood function and wonder where to plug it in,
        #       pass a custom ln_Z0 array in the kwargs!
        ln_Z0_arr = get_zero_evidence(data, rms, normalize=normalize)

    for y, x in np.ndindex(shape):
        ln_Z0 = [ln_Z0_arr[y, x], np.nan] if 0 in peaks else None
        Z_xy = lnZ_xy(x=x, y=y, peaks=peaks, lnZ0=ln_Z0, **kwargs)
        for z_idx, (_, lnZ) in enumerate(zip(peaks, Z_xy)):
            lnZs[z_idx, y, x] = lnZ.n
            err_lnZs[z_idx, y, x] = lnZ.std_dev

    header = _tinker_header(header, ctype3='BAYESIAN EVIDENCE',
                            bunit='ln(Z_npeaks)')

    for p in peaks:
        header['PLANE{}'.format(p + 1)] = p

    hdu = fits.PrimaryHDU(np.vstack([lnZs, err_lnZs]), header)

    if writeto:
        hdu.writeto(writeto, overwrite=True)

    return hdu


def pars_xy(x, y, mode='fast', stat='mle', **kwargs):
    a = analyzer_xy(x=x, y=y, **kwargs)
    try:
        if mode == 'slow':
            pars = a.get_best_fit()['parameters']
            errs = a.get_stats()['modes'][0]['sigma'] # Using symetrical sigma
        elif mode == "fast":
            pars, errs = get_stats_fast(a, stat=stat)
        else:
            raise ValueError("mode should be one of the ['slow', 'fast']")
    except IOError:
        return np.nan, np.nan
    return pars, errs


def parcube(shape, npeaks, npars, origin=(0, 0),
            header=None, writeto=None, **kwargs):
    """
    Construct a fits HDU with best fit parameters at all xy
    positions for a cube of a given shape.
    Optionally, writes a fits file.

    Additional keyword args are passed to analyzer_xy function.
    """
    # TODO: oh so much! get MLE, MAP, or mean at the posterior?
    #       errors, and how to choose errors, and whether they
    #       should be symmetrical or asymmetrical...

    if origin != (0, 0):
        # TODO: implement this
        raise NotImplementedError("wip")

    parcube = np.empty(shape=(npars * npeaks, ) + shape)
    parcube.fill(np.nan)
    errcube = parcube.copy()

    for y, x in np.ndindex(shape):
        pars, errs = pars_xy(x=x, y=y, npars=npars, npeaks=npeaks, **kwargs)
        parcube[:, y, x] = pars
        print(parcube[:, y, x])
        # FIXME: get the confidence intervals on pars as well
        print(errs)
        errcube[:, y, x] = errs

    header = _tinker_header(header, ctype3='BEST FIT PARAMETERS',
                            bunit='VARIOUS')

    hdu = fits.PrimaryHDU(np.vstack([parcube, errcube]), header)

    # TODO: I think it's better to take in ammonia model...
    #       the names of parameters are already there, then I
    #       can also put prior information along in a cube

    if writeto:
        hdu.writeto(writeto, overwrite=True)

    return hdu


def bayesian_npeaks_averaging(zarr, npeaks_max, header=None, writeto=None,
                              force_float128=True):
    """
    Calculates an image of number of components per pixel via a simplified
    version of Bayesian averaging. Basically, a number of components weighted
    by the Bayesian evidence.

    Parameters
    ----------
    zarr : an np.ndarray of log(Z) images

    npeaks : int; how many multicomponent models to average on

    header : if saving to a file, the header to use for FITS image

    writeto : output FITS file (optional)

    force_float128 : bool; whether to force intermediate arrays to float128

    Returns
    -------
    stat : ndims-shaped array for MLE or MAP, ndims x 2 shape for mean
           statistics (for mean and sigma values)

    """
    dtype = np.float128 if force_float128 else zarr.dtype

    # we are computing intermediate values of bayesian evidence from its log,
    # so in cases of "extreme" detection with very high S/N this might overflow
    # ... so for now let's raise overflow errors
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            # because zarr is actually ln(P(M | D)), remember?
            pzarr = np.exp(zarr, dtype=dtype)
            npeaks_map = (pzarr[:npeaks_max + 1]
                          * np.arange(npeaks_max + 1)[:, None, None]).sum(
                          axis=0) / pzarr[:npeaks_max + 1].sum(axis=0)
            # why change it back? some libraries aren't very friendly to
            # float128 types, e.g. matplotlib's imshow doesn't plot them
            npeaks_map = npeaks_map.astype(zarr.dtype)
        except FloatingPointError:
            log.warn("Exponential overflow detected. Will loop over pixels"
                     " using the `bigfloat` package framework instead for"
                     " arbitrary precision. It's slower and bigfloat package"
                     " is required.")
            import bigfloat
            npeaks_map = np.full_like(zarr[0], np.nan)
            for (y, x) in np.ndindex(zarr[0].shape):
                lnZs_xy = zarr[:npeaks_max + 1, y, x]
                numerator = np.sum([i * bigfloat.exp(lnZ)
                                    for (i, lnZ) in enumerate(lnZs_xy)])
                denumerator = np.sum([bigfloat.exp(lnZ)
                                      for lnZ in lnZs_xy])

                npeaks_map[y, x] = np.float64(numerator / denumerator)

    if writeto:
        if header:
            header = _tinker_header(header, bunit='', flatten=True)
        hdu = fits.PrimaryHDU(npeaks_map, header)
        hdu.writeto(writeto, overwrite=True)

    return npeaks_map
