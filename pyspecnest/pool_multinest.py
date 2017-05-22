"""
Instead of MPI parallelisation for each pixel, it is [citation needed] faster
to simply distribute pixels to different processes via pooling. The speedup is
not about the actual sampling, but the overheads are only executed once...
"""
from __future__ import division
from collections import OrderedDict
from astropy.io import fits
from astropy import log
import multiprocessing
import numpy as np
import subprocess
import time
import sys

def get_log_file_fmatter(log_file=None, prefix='g35-nh3_',
                         proj_dir = 'Projects/g35-vla-nh3/'):
    """ Backwards compatibility with g35-vla-nh3 repo. """
    if log_file is None:
        from go_home import home_dir
        log_dir = home_dir + proj_dir + 'nested-sampling/logs/'
        log_file_fmatter = log_dir + 'g35-nh3_x{}y{}-npeaks{}.log'

    return log_file_fmatter

def get_logger(x, y, npeaks, log_file_fmatter=None):
    """ One log file per pixel to avoid race condition """
    if log_file_fmatter is None:
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
    return subprocess.call(cmd.split(' '),
                           stdout = stdout,
                           shell = False)

def get_xy_sorted(arr, xy_indices=None, cut = None):
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
    if snr11 is None or snr22 is None:
        snr11, snr22 = get_vla_snr()
    line_to_snr = {'nh311': snr11, 'nh322': snr22}

    return get_xy_sorted(line_to_snr[line], np.indices(snr11.shape), **kwargs)

def bayes_factor_order(kfile='Ks.fits', idx=0, **kwargs):
    K = fits.getdata(kfile)[idx]

    return get_xy_sorted(K, np.indices(K.shape), **kwargs)

def get_vla_snr():
    """
    Backwards compatibility with the g35-vla-nh3 repo.
    """
    from opencube import make_cube_shh
    cubes = make_cube_shh() # comes with pregen snr attributes...
    snrmap11, snrmap22 = cubes.snr11, cubes.snr22

    return snrmap11, snrmap22

def xy_sorted_by(method='Bfactor', **kwargs):
    """
    Returns xy indices sorted according to a given method.
    Additional keyword arguments are passed to the method f-ions.
    """
    method_to_func = {'Bfactor': bayes_factor_order,
                      'snr'    : snr_order          }

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

    cmd_string = ("{}python3 {} {} ".format(prefix, script, npeaks)
                  + "{} {} 0 {}")
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

if __name__ == '__main__':
    """ Example run, custom-tailored to the VLA data on G035.39. """
    # NOTE: normal dict would mess up the order of the arguments
    default_args = OrderedDict([('npeaks', 1), ('method', "snr"), ('cut', 8),
                                ('n_cpu', 7)])

    runtime_args = {}
    for i, (argname, argval) in enumerate(default_args.items()):
        runtime_args[argname] = try_get_args(i, argval, type(argval))

    method = runtime_args.pop('method')
    n_cpu = runtime_args.pop('n_cpu')
    if method == 'snr':
        line = 'nh322'
        snrmap11, snrmap22 = get_vla_snr()

        tasklist_kwargs=dict(n_cpu=n_cpu, method='snr', line=line,
                             snr11=snrmap11, snr22=snrmap22)

    if method == 'Bfactor':
        # TODO FIXME:
        raise NotImplementedError("What used to work, no longer does. "
                                  "Don't go berserk, put blame on Vlas.")

    tasklist_kwargs.update(runtime_args)

    tasks = get_tasks(**tasklist_kwargs)

    pool = multiprocessing.Pool(processes=n_cpu)
    pool.map(work, tasks)

def testing_K_sort(Kfile='Ks.fits', index=0, debug=False):
    if debug:
        log.setLevel('DEBUG')

    K_vals = fits.getdata(Kfile)[index]
    K_new = np.inf

    tasks = get_tasks(method='Bfactor', npeaks=1, cut=20)
    for job in tasks:
        _, _, npeaks, y, x, _, p = job.split(' ')
        x, y = int(x), int(y)
        K_new, K_old = K_vals[y, x], K_new

        assert K_old > K_new

        log.debug("K = {:7.2f} at (x, y) = ({:2d}, "
                  "{:2d}), {} done".format(K_new, x, y, p))

def testing_snr_sort(snrmap11=None, snrmap22=None, debug=False,
                     cut=5, line='nh311', n_cpu=7, run=False):
    """
    Assures that the S/N ordering is being executed properly.
    Was written for an ammonia cube data, have yet to generalize.
    """
    if debug:
        log.setLevel('DEBUG')

    if snrmap11 is None and snrmap22 is None:
        snrmap11, snrmap22 = get_vla_snr()

    snr, snr_prev = {}, {'nh311': np.inf, 'nh322': np.inf}
    tasks_by_snr = get_tasks(n_cpu=n_cpu, method='snr', cut=cut, line=line,
                             snr11=snrmap11, snr22=snrmap22, testing=True)
    for job in tasks_by_snr:
        _, _, _, npeaks, y, x, _, p = job.split(' ')
        x, y = int(x), int(y)
        snr['nh311'], snr['nh322'] = snrmap11[y, x], snrmap22[y, x]

        # make sure the snr job list progresses downwards
        assert snr[line] <= snr_prev[line]

        log.debug("S/R @ NH3 (1,1) = {:.2f}, "
                  "S/R @ NH3 (2,2) = {:.2f} at (x, y) = "
                  "({:2d}, {:2d}), {} done".format(snr['nh311'], snr['nh322'],
                                                   x, y, p))

        # used later for recurrent relation reasons...
        snr_prev['nh311'], snr_prev['nh322'] = snr['nh311'], snr['nh322']

    if run:
        pool = multiprocessing.Pool(processes=n_cpu)
        pool.map(work, tasks_by_snr)
