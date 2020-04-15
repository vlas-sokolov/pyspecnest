"""
Instead of MPI parallelisation for each pixel, it is [citation needed] faster
to simply distribute pixels to different processes via pooling. The speedup is
not about the actual sampling, but the overheads are only executed once...
"""

from __future__ import division
import os
import sys
import time
import errno
import subprocess
import numpy as np
from astropy.io import fits
from astropy import log


class Config:
    """ Global configuration attributes """
    log_file_fmatter = "multinest-x{}y{}-npeaks{}.log"
    log_dir = None
    log_dir_default_relative = "multinest-logs"


def get_log_file_fmatter():
    """ Returns a logging file path formatter """
    if not Config.log_dir:
        log_dir = Config.log_dir_default_relative
    else:
        log_dir = Config.log_dir

    try:
        os.makedirs(log_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    log_file_fmatter = os.path.join(log_dir, Config.log_file_fmatter)

    return log_file_fmatter


def get_logger(x, y, npeaks):
    """ One log file per pixel to avoid race condition """
    log_file_fmatter = get_log_file_fmatter()
    log_file = log_file_fmatter.format(x, y, npeaks)

    return open(log_file, "w")


def work(cmd):
    """ Hacks around arg-feeding the shell while neatly redirecting stdout """
    try:
        _, _, npeaks, y, x, _, perc = cmd.split(' ')
    except ValueError:
        # there's an `echo` command in front of cmd,
        # so we're runnning in `testing=True` mode
        _, _, _, npeaks, y, x, _, perc = cmd.split(' ')
    stdout = get_logger(x, y, npeaks)
    # testing:
    time_string = time.strftime("%Y-%m-%d %H:%M")
    log.info("{}: processing (x, y) = ({:2d}, {:2d});"
             " {} done".format(time_string, int(x), int(y), perc))

    # hack to get the total number of jobs
    cmd = ' '.join(cmd.split(' ')[:-1])

    # execute the worker
    return subprocess.call(cmd.split(' '), stdout=stdout, shell=False)


def get_xy_sorted(arr, xy_indices=None, cut=None):
    """
    Sort xy indices by a value array.
    Adapted from pyspeckit's SpectralCube.py.
    """
    if xy_indices is None:
        xy_indices = np.indices(arr.shape)
    yy, xx = xy_indices
    arrsort = np.argsort((1 / (arr - np.nanmin(arr) + .1)).flat)

    mask = np.isfinite(arr).flat
    if cut:
        mask = mask & (np.nan_to_num(arr.flat) > cut)

    sorted_arr = list(zip(xx.flat[arrsort][mask[arrsort]],
                          yy.flat[arrsort][mask[arrsort]]))
    return sorted_arr


def snr_order(line='nh311', snr11=None, snr22=None, **kwargs):
    """
    Was written for an ammonia cube data, have yet to generalize.
    """
    line_to_snr = {'nh311': snr11, 'nh322': snr22}

    return get_xy_sorted(line_to_snr[line], np.indices(snr11.shape), **kwargs)


def bayes_factor_order(kfile='Ks.fits', idx=0, **kwargs):
    K = fits.getdata(kfile)[idx]

    return get_xy_sorted(K, np.indices(K.shape), **kwargs)


def xy_sorted_by(method='Bfactor', **kwargs):
    """
    Returns xy indices sorted according to a given method.
    Additional keyword arguments are passed to the method f-ions.
    """
    method_to_func = {'Bfactor': bayes_factor_order,
                      'snr': snr_order}

    return method_to_func[method](**kwargs)


def perc(i, n_jobs, n_cpu, split=False):
    if split:
        jobs_per_cpu = n_jobs / n_cpu
        p = (i % jobs_per_cpu) / jobs_per_cpu * 100
    else:
        p = i / n_jobs * 100
    return "%{:.2f}".format(p)


def get_tasks(n_cpu, npeaks=1, method='Bfactor', testing=False,
              script="innocent_script.py", xy_order=None, **kwargs):
    if xy_order is None:
        xy_order = xy_sorted_by(method, **kwargs)

    prefix = 'echo ' if testing else ''

    cmd_string = ("{}python{} {} {} ".format(prefix, sys.version_info.major,
                                             script, npeaks) + "{} {} 0 {}")
    n_jobs = len(xy_order)
    tasks = [cmd_string.format(y, x, perc(i, n_jobs, n_cpu))
             for i, (x, y) in enumerate(xy_order)]

    return tasks


def try_get_args(n, fallback, forcetype=str):
    try:
        # sys.argv[0] is some env executable path...
        arg = forcetype(sys.argv[n+1])
    except IndexError:
        arg = fallback

    return arg
