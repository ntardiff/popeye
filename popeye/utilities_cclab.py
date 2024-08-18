"""This module contains various utility methods that support functionality in
other modules.  The multiprocessing functionality also exists in this module,
though that might change with time.

"""

from __future__ import division
import sys, os, time, fnmatch, copy, ctypes
from multiprocessing import Array
from itertools import repeat
from random import shuffle
import datetime

import numpy as np
import nibabel
from scipy.stats import gamma
from scipy.signal import detrend
from scipy.optimize import brute, fmin_powell, fmin, least_squares, minimize, differential_evolution
# from scipy.stats import linregress
from scipy.integrate import romb, trapz
# from scipy import c_, ones, dot, stats, diff
# from scipy.linalg import inv, solve, det, norm
# from numpy import log, pi, sqrt, square, diagonal
# from numpy.random import randn, seed
import sharedmem
from numba import jit, types

# Python 3 compatibility below:
try:  # pragma: no cover
    import cPickle
except ImportError:  # pragma: no cover
    import _pickle as cPickle

try:  # pragma: no cover
    from types import SliceType
except ImportError:  # pragma: no cover
    SliceType = slice

try: # pragma: no cover
    xrange
except NameError:  # pragma: no cover
    xrange = range
    
import multiprocessing as mp
if mp.get_start_method() != 'fork':
    mp.set_start_method('fork',force=True)


def regularizing_error_function(parameter, bundle, p_bounds, thr=0.10): # pragma: no cover
    
    # if out of bounds
    if parameter < p_bounds[0] or parameter > p_bounds[1]:
            return np.inf
    
    # execute
    output = regularizing_objective_function(parameter, bundle)
    
    # what?
    error = np.mean([o.rss for o in output if not np.isnan(o.rss) and o.rsquared > thr])
    
    # when?
    now = str(datetime.datetime.now()).replace(" ","_").replace(".","_").replace(":","_").replace('-','_')
    
    # verbose!
    print('parameter=%.05f    error=%.05f    now=%s' %(o.model.parameter, error, now))
    
    return error

def regularizing_objective_function(parameter, bundle): # pragma: no cover
    
    # attach the guess for tau to each of the voxels in the bundle
    for voxel in bundle:
        model = voxel[1]
        model.parameter = parameter
        
    # fit each of the voxels
    num_cpus = sharedmem.cpu_count()-1
    with sharedmem.Pool(np=num_cpus) as pool:
        output = pool.map(parallel_fit, bundle)
    
    return output

def regularizer(bundle, p_grid, p_bounds, Ns=None): # pragma: no cover
    
    p0 = brute(regularizing_error_function, (p_grid,), args=(bundle, p_bounds), Ns=Ns, finish=None, full_output=True)
    phat = fmin_powell(regularizing_error_function, p0[0], args=(bundle, p_bounds), xtol=1e-4, ftol=1e-4, full_output=True, retall=True)
    
    return p0,phat

def _gamma_difference_hrf(tr, oversampling=1, time_length=32., onset=0.,
                         delay=5, undershoot=15., dispersion=1.,
                         u_dispersion=1., ratio=0.167):
    """ Compute an hrf as the difference of two gamma functions
    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the hrf
    Returns
    -------
    hrf: array of shape(length / tr * oversampling, float),
         hrf sampling on the oversampled time grid
    """
    dt = tr / oversampling
    time_stamps = np.linspace(0, time_length, int(float(time_length) / dt))
    time_stamps -= onset / dt
    hrf = gamma.pdf(time_stamps, delay / dispersion, dt / dispersion) - \
        ratio * gamma.pdf(
        time_stamps, undershoot / u_dispersion, dt / u_dispersion)
    hrf /= trapz(hrf)
    return hrf

def spm_hrf(delay, tr, oversampling=1, time_length=32., onset=0.):
    """ Implementation of the SPM hrf model
    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the response
    Returns
    -------
    hrf: array of shape(length / tr * oversampling, float),
         hrf sampling on the oversampled time grid
    """
    return _gamma_difference_hrf(tr, oversampling, time_length, onset, delay=5+delay, undershoot=15+delay,)


def glover_hrf(delay, tr, oversampling=1, time_length=32., onset=0.):
    """ Implementation of the Glover hrf model
    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the response
    Returns
    -------
    hrf: array of shape(length / tr * oversampling, float),
         hrf sampling on the oversampled time grid
    """
    return _gamma_difference_hrf(tr, oversampling, time_length, onset,
                                delay=5+delay, undershoot=15+delay, dispersion=.9,
                                u_dispersion=.9, ratio=.35)


def grid_slice(start, stop, Ns, dryrun=False):
    
    #### NOTE: there is some weird stuff going on when Ns is => stop-start
    #### it will return an array/slice object that is longer and wide by 1
    #### element. This is easy to fix in the dry-run mode but not so with
    #### the slice object. Why oh why does scipy.brute insist on slice?
    
    # special case
    if Ns == 2:
        step = stop-start
    # all others
    else:
        step = np.diff(np.linspace(start,stop,Ns))[0]
    
    # if true, this return the ndarray rather than slice object.
    if dryrun: # pragma: no cover
        return arange(start, stop+step, step) # pragma: no cover
    else:
        return slice(start, stop+step, step)
        
def distance_mask(x, y, sigma, deg_x, deg_y, amplitude=1):
    
    distance = (deg_x - x)**2 + (deg_y - y)**2
    mask = np.zeros_like(distance, dtype='uint8')
    mask[distance < sigma**2] = 1
    mask *= amplitude
    
    return mask


@jit(nopython=True,parallel=False)
def generate_og_receptive_field_2d(x,y,sigma,deg_x,deg_y):
    
    d = (deg_x-x)**2 + (deg_y-y)**2
        
    rf = np.exp(-d / (2.0 * sigma**2))
    
    return rf

def generate_og_receptive_field(x,y,sigma,deg_x,deg_y):
    #return rf in 1D for matrix multiplication. Could     
        
    rf = generate_og_receptive_field_2d(x,y,sigma,deg_x,deg_y)
    
    return np.reshape(rf,(rf.shape[0]*rf.shape[1],1)).astype(np.float32)

def stim2d(stim_arr):
    stim_arr_long = stim_arr.transpose(2,0,1)
    stim_arr_long = np.reshape(stim_arr_long,(stim_arr_long.shape[0],-1))
    
    return stim_arr_long


# @jit(nopython=True,parallel=False)
# def generate_rf_timeseries_base(stim_arr,rf):
#     return np.dot(stim_arr,rf)


#have not found that this benefits from numba on local machine...
def generate_rf_timeseries(stim_arr,rf):
    #return np.squeeze(generate_rf_timeseries_base(stim_arr,rf))
    return np.squeeze(np.dot(stim_arr,rf))
    #return np.squeeze(np.matmul(stim_arr,rf))
    #return np.squeeze(stim_arr @ rf)
    #ts = np.squeeze(stim_arr @ rf)
    #ts[ts==0] = np.finfo(np.float32).tiny #can get into trouble w/ optimization w/ 0s
    #return ts


def recast_estimation_results(output, grid_parent, overloaded=False):
    
    # load the gridParent
    dims = list(grid_parent.shape)
    dims = dims[0:3]
    
    if overloaded == True and output[0].overloaded_estimate is not None:
        dims.append(len(output[0].overloaded_estimate)+1)
    else:
        dims.append(len(output[0].estimate)+1)
        
    # initialize the statmaps
    estimates = np.zeros(dims)
    
    # extract the prf model estimates from the results queue output
    for fit in output:
        
        if not np.isnan(fit.rsquared): # pragma: no cover
        
            # gather the estimate + stats
            if overloaded == True and fit.overloaded_estimate is not None:
                voxel_dat = list(fit.overloaded_estimate)
            else:
                voxel_dat = list(fit.estimate)
                
            voxel_dat.append(fit.rsquared)
            voxel_dat = np.array(voxel_dat)
            
            # assign to
            estimates[tuple(fit.voxel_index)] = voxel_dat
            
    # get header information from the gridParent and update for the prf volume
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    hdr.set_data_shape(dims)
    
    # recast as nifti
    nifti_estimates = nibabel.Nifti1Image(estimates,aff,header=hdr)
    
    return nifti_estimates

def recast_xval_results(output, bootstraps, indices, grid_parent, overloaded=False, ncpus=30):

    # load the grid_parent (x,y,z)
    dims = list(grid_parent.shape)
    dims = dims[0:3]
    
    # params + rsquared + cod
    if overloaded == True and output[0].overloaded_estimate is not None:
        dims.append(len(output[0].overloaded_estimate)+2)
    else:
        dims.append(len(output[0].estimate)+2)
    
    # add bootstrap dim
    dims.append(bootstraps)
    
    # initialize the statmaps
    estimates = generate_shared_array(np.zeros(dims), ctypes.c_double)
    
    # parallelizer
    def parallel_loader(index): # pragma: no cover
        
        # gather up fits for this voxel
        fits = [o for o in output if list(o.voxel_index) == list(index)]
        
        # gather the estimate + stats
        if overloaded == True and fits[0].overloaded_estimate is not None:
            params = np.array([fit.overloaded_estimate for fit in fits])
        else:
            params = np.array([fit.estimate for fit in fits])
            
        # xval metrics
        rsq = np.array([fit.rsquared for fit in fits])
        cod = np.array([fit.cod for fit in fits])
            
        # assign
        estimates[index[0],index[1],index[2]] = np.concatenate((params, cod[:,np.newaxis], rsq[:,np.newaxis]),-1).T
        
        return None
        
    # populate data structure
    with sharedmem.Pool(np=ncpus) as pool:
        pool.map(parallel_loader, indices)
        
    # header & affine
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    hdr.set_data_shape(dims)
    
    # recast as nifti
    nifti_estimates = nibabel.Nifti1Image(estimates,aff,header=hdr)
    
    return nifti_estimates

def make_nifti(data, grid_parent=None):
    
    if grid_parent:
        
        # get header information from the gridParent and update for the prf volume
        aff = grid_parent.get_affine()
        hdr = grid_parent.get_header()
        
        # recast as nifti
        nifti = nibabel.Nifti1Image(data,aff,header=hdr)
        
    else:
        aff = np.eye(4,4)
        nifti = nibabel.Nifti1Image(data,aff)
        
    return nifti

def generate_shared_array(unshared_arr,dtype):

    r"""Creates synchronized shared arrays from numpy arrays.

    The function takes a numpy array `unshared_arr` and returns a shared
    memory object, `shared_arr`.  The user also specifies the data-type of
    the values in the array with the `dataType` argument.  See
    multiprocessing.Array and ctypes for details on shared memory arrays and
    the data-types.

    Parameters
    ----------
    unshared_arr : ndarray
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    dtype : ctypes instance
        The data-type specificed has to be an instance of the ctypes library.
        See ctypes for details.

    Returns
    -------
    shared_arr : synchronized shared array
        An array that is read accessible from multiple processes/threads.
    """

    shared_arr = sharedmem.empty(unshared_arr.shape, dtype=dtype)
    shared_arr[:] = unshared_arr[:]
    return shared_arr

# normalize to a specific range
def normalize(array, imin=-1, imax=1, axis=-1):

    r"""A short-hand function for normalizing an array to a desired range.

    Parameters
    ----------
    array : ndarray
        An array to be normalized.

    imin : float
        The desired minimum value in the output array.  Default: -1

    imax : float
        The desired maximum value in the output array.  Default: 1


    Returns
    -------
    array : ndarray
        The normalized array with imin and imax as the minimum and
        maximum values.
    """

    new_arr = array.copy()

    if np.ndim(imin) == 0:
        dmin = new_arr.min()
        dmax = new_arr.max()
        new_arr = new_arr-dmin
        new_arr = new_arr*(imax - imin)
        new_arr = new_arr/(dmax - dmin)
        new_arr = new_arr+imin
    else:
        dmin = new_arr.min(axis=axis)
        dmax = new_arr.max(axis=axis)
        new_arr -= dmin[:,np.newaxis]
        new_arr *= imax[:,np.newaxis] - imin[:,np.newaxis]
        new_arr /= dmax[:,np.newaxis] - dmin[:,np.newaxis]
        new_arr += imin[:,np.newaxis]

    return new_arr


# generic gradient descent
def gradient_descent_search(data, error_function, objective_function, parameters, bounds, verbose,**kwargs):

    r"""A generic gradient-descent error minimization function.

    The values inside `parameters` are used as a seed-point for a
    gradient-descent error minimization procedure [1]_.  The user must
    also supply an `objective_function` for producing a model prediction
    and an `error_function` for relating that prediction to the actual,
    measured `data`.

    In addition, the user may also supply  `fit_bounds`, containing pairs
    of upper and lower bounds for each of the values in parameters.
    If `fit_bounds` is specified, the error minimization procedure in
    `f` will return an Inf whether the parameters exceed the
    minimum or maxmimum values specified in `fit_bounds`.


    Parameters
    ----------
    parameters : tuple
        A tuple of values representing a model setting.

    args : tuple
        Extra arguments to `objective_function` beyond those in `parameters`.

    fit_bounds : tuple
        A tuple containing the upper and lower bounds for each parameter
        in `parameters`.  If a parameter is not bounded, simply use
        `None`.  For example, `fit_bounds=((0,None),(-10,10),)` would
        bound the first parameter to be any positive number while the
        second parameter would be bounded between -10 and 10.

    data : ndarray
        The actual, measured time-series against which the model is fit.

    error_function : callable
        The error function that relates the model prediction to the
        measure data.  The error function returns a float that represents
        the residual sum of squared errors between the prediction and the
        data.

    objective_function : callable
        The objective function that takes `parameters` and `args` and
        proceduces a model time-series.

    Returns
    -------
    estimate : tuple
        The model solution given `parameters` and `objective_function`.


    References
    ----------

    .. [1] Fletcher, R, Powell, MJD (1963) A rapidly convergent descent
    method for minimization, Compututation Journal 6, 163-168.

    """
    
    # if bounds is None:
    #     output = least_squares(error_function_residual, parameters, method='lm',
    #                            args=(data, objective_function, verbose))
    # else:
    #     output = least_squares(error_function_residual, parameters, bounds=bounds,
    #                            args=(data, objective_function, verbose))
    output = minimize(error_function, parameters, bounds=bounds, method='trust-constr', #method='COBYLA', #method='SLSQP', #method='COBYLA',
                      args=(data, objective_function, verbose),**kwargs)

    return output


def global_search(data, error_function, objective_function, bounds, verbose,
                  max_bounds=[-100,100], **kwargs):

    
    #minimize accepts None/inf as a bound, but differential evolution doesn't
    bounds = np.array(bounds)
    bounds[bounds[:,0]==None,0] = max_bounds[0]
    bounds[bounds[:,1]==None,1] = max_bounds[1]
    bounds = tuple(map(tuple,bounds)) #back to tuple...maybe not required    

    output = differential_evolution(error_function, bounds=bounds, 
                      args=(data, objective_function, verbose,),**kwargs)

    return output

def make_dmat(ts):
    return np.column_stack((ts,np.ones(ts.shape,dtype=np.float32)))

def lsq(x,y):
    dmat = make_dmat(x)
    
    return np.linalg.lstsq(dmat, y, rcond=None)


@jit(nopython=True, parallel=False)
def stacker(x,y):
    return np.hstack((x,y))

@jit(nopython=True, parallel=False)
def rss(data,prediction):
    d = data - prediction
    return d.dot(d)
    #return np.nansum((data-prediction)**2)

@jit(nopython=True, parallel=False)
def residual(data,prediction):
    return (data-prediction)**2

@jit(nopython=True, parallel=False)
def check_parameters(parameters, bounds):
    for i in range(len(parameters)):
        b = bounds[i]
        p = parameters[i]
        if b[0] and p < b[0]:
            return None
        if b[1] and b[1] < p:
            return None
    
    # merge the parameters and arguments
    ensemble = []
    ensemble.extend(parameters)
    return ensemble

# def error_function_rss(parameters, data, objective_function, verbose):
#     prediction = objective_function(*parameters)
#     error = rss(data, prediction)
#     return error

def error_function_rss(parameters, data, objective_function,verbose):
    prediction = objective_function(*parameters)
    d = data - prediction
    error = d.dot(d)
    #error = np.sum((data-prediction)**2)
    
    #return something very large if we encounter bad values
    if np.isfinite(error):
        return error
    else:
        d = data - data.mean()
        return d.dot(d)*1e10
    
    
@jit(nopython=True,parallel=False)
def do_lsq_error(prediction, data, rawrss):
    
    #minimize algorithms don't obey constraints no matter how hard I try...
    if np.allclose(prediction,0):
        return rawrss*1e10
    
    dmat = np.column_stack((prediction,np.ones(prediction.shape,dtype=np.float32)))
    try:
        sol = np.linalg.lstsq(dmat,data)
    except Exception:
        #return something very large if we encounter bad values
        return rawrss*1e10
    error = sol[1][0]
    
    if sol[0][0] >= 0: #check beta for constraint
        return error
    else:
        #given our regression equation, the minimial constrained least squares solution is 
        #beta = 0, intercept = mean...e.g. the error is the rss of the data
        return rawrss - sol[0][0]
        #d = data - data.mean()
        #return d.dot(d) - sol[0][0] # let's help out gradient w/ L1 penality on negative beta
    
    # if np.isfinite(error):
    #     if sol[0][0] >= 0: #check beta for constraint
    #         return error
    #     else:
    #         #given our regression equation, the minimial constrained least squares solution is 
    #         #beta = 0, intercept = mean...e.g. the error is the rss of the data
    #         return rawrss - sol[0][0]
    #         #d = data - data.mean()
    #         #return d.dot(d) - sol[0][0] # let's help out gradient w/ L1 penality on negative beta
    # else:
    #     return rawrss*1e10
    

@jit(nopython=True,parallel=False)    
def dist_con(x):
    return np.sqrt(x[0]**2 + x[1]**2)
    
@jit(nopython=True,parallel=False)   
def prfsize_con(x,outer_limit):
    return np.sqrt(x[0]**2 + x[1]**2) - outer_limit*x[2]
    
def error_function_lsq(parameters, data, objective_function,verbose):
    
    prediction = objective_function(*parameters)
    dmat = np.column_stack((prediction,np.ones(prediction.shape)))
    sol = np.linalg.lstsq(dmat,data,rcond=None)
    error = sol[1][0]
    #return something very large if we encounter bad values
    
    if np.isfinite(error):
        if sol[0][0] >= 0: #check beta for constraint
            return error
        else:
            #given our regression equation, the minimial constrained least squares solution is 
            #beta = 0, intercept = mean...e.g. the error is the rss of the data
            d = data - data.mean()
            return d.dot(d) - sol[0][0] # let's help out gradient w/ L1 penality on negative beta
    else:
        d = data - data.mean()
        return d.dot(d)*1e10

# generic error function
def error_function_residual(parameters, data, objective_function, verbose):
    prediction = objective_function(*parameters)
    error = residual(data, prediction)
    return error

def brute_force_search(data, error_function, objective_function, grids, Ns=None, verbose=False):

    r"""A generic brute-force grid-search error minimization function.

    The user specifies an `objective_function` and the corresponding
    `args` for generating a model prediction.  The `brute_force_search`
    uses `search_bounds` to dictate the bounds for evenly sample the
    parameter space.

    In addition, the user must also supply  `fit_bounds`, containing pairs
    of upper and lower bounds for each of the values in parameters.
    If `fit_bounds` is specified, the error minimization procedure in
    `error_function` will return an Inf whether the parameters exceed the
    minimum or maxmimum values specified in `fit_bounds`.

    The output of `brute_force_search` can be used as a seed-point for
    the fine-tuned solutions from `gradient_descent_search`.


    Parameters
    ----------
    args : tuple
        Arguments to `objective_function` that yield a model prediction.

    grids : tuple
        A tuple indicating the search space for the brute-force grid-search.
        The tuple contains pairs of upper and lower bounds for exploring a
        given dimension.  For example `grids=((-10,10),(0,5),)` will
        search the first dimension from -10 to 10 and the second from 0 to 5.
        These values cannot be `None`.

        For more information, see `scipy.optimize.brute`.

    bounds : tuple
        A tuple containing the upper and lower bounds for each parameter
        in `parameters`.  If a parameter is not bounded, simply use
        `None`.  For example, `fit_bounds=((0,None),(-10,10),)` would
        bound the first parameter to be any positive number while the
        second parameter would be bounded between -10 and 10.

    Ns : int
        Number of samples per stimulus dimension to sample during the ballpark search.

        For more information, see `scipy.optimize.brute`.

    data : ndarray
       The actual, measured time-series against which the model is fit.

    error_function : callable
       The error function that relates the model prediction to the
       measure data.  The error function returns a float that represents
       the residual sum of squared errors between the prediction and the
       data.

    objective_function : callable
      The objective function that takes `parameters` and `args` and
      proceduces a model time-series.

    Returns
    -------
    estimate : tuple
       The model solution given `parameters` and `objective_function`.

    """
    
    # if user provides their own grids
    if isinstance(grids[0], SliceType):
        output = brute(error_function,
                       args=(data, objective_function, verbose),
                       ranges=grids,
                       finish=None,
                       full_output=True,
                       disp=False)

    # otherwise specify (min,max) and Ns for each dimension
    else:
        output = brute(error_function,
               args=(data, objective_function, verbose),
               ranges=grids,
               Ns=Ns,
               finish=None,
               full_output=True,
               disp=False)
    return output

# generic error function
def error_function(parameters, bounds, data, objective_function, verbose):

    r"""A generic error function with bounding.

    Parameters
    ----------
    parameters : tuple
        A tuple of values representing a model setting.

    args : tuple
        Extra arguments to `objective_function` beyond those in `parameters`.

    data : ndarray
       The actual, measured time-series against which the model is fit.

    objective_function : callable
        The objective function that takes `parameters` and `args` and
        proceduces a model time-series.

    debug : bool
        Useful for debugging a model, will print the parameters and error.

    Returns
    -------
    error : float
        The residual sum of squared errors between the prediction and data.
    """
    
    ############
    #   NOTE   #
    ############
    # as of now, this will not work if your model has 1 parameter.
    # i think it is because scipy.optimize.brute returns
    # a scalar when num params is 1, and a tuple/list
    # when num params is > 1. have to look into this further
    
    # check if parameters are inside bounds
    for p, b in zip(parameters,bounds):
        # if not return an inf
        if b[0] and p < b[0]:
            return np.inf
        if b[1] and b[1] < p:
            return np.inf
            
    # merge the parameters and arguments
    ensemble = []
    ensemble.extend(parameters)
    
    # compute the RSS
    prediction = objective_function(*ensemble)
    
    # if nan, return inf
    if np.any(np.isnan(prediction)):
        return np.inf # pragma: no cover
        
    # else, return RSS
    error = np.nansum((data-prediction)**2)
    # error = norm(data-prediction)
    
    # print for debugging
    if verbose:
        print(parameters, error)

    return error

def double_gamma_hrf(delay, tr, fptr=1.0, integrator=trapz,dtype='float32'):

    r"""The double gamma hemodynamic reponse function (HRF).
    The user specifies only the delay of the peak and undershoot.
    The delay shifts the peak and undershoot by a variable number of
    seconds. The other parameters are hardcoded. The HRF delay is
    modeled for each voxel independently. The form of the HRF and the
    hardcoded values are based on previous work [1]_.

    Parameters
    ----------
    delay : float
        The delay of the HRF peak and undershoot.

    tr : float
        The length of the repetition time in seconds.

    fptr : float
        The number of stimulus frames per reptition time.  For a
        60 Hz projector and with a 1 s repetition time, the fptr
        would be equal to 60.  It is possible that you will bin all
        the frames in a single TR, in which case fptr equals 1.

    integrator : callable
        The integration function for normalizing the units of the HRF
        so that the area under the curve is the same for differently
        delayed HRFs.  Set integrator to None to turn off normalization.

    Returns
    -------
    hrf : ndarray
        The hemodynamic response function to convolve with the stimulus
        timeseries.

    Reference
    ----------
    .. [1] Glover, GH (1999) Deconvolution of impulse response in event related
    BOLD fMRI. NeuroImage 9, 416-429.

    """
    from scipy.special import gamma
    
    # add delay to the peak and undershoot params (alpha 1 and 2)
    alpha_1 = float(5 + delay)
    beta_1 = 1.0
    c = 0.1
    alpha_2 = float(15 + delay)
    beta_2 = 1.0
    
    t = np.arange(0,32,tr)
    
    hrf = ( ( ( t ** (alpha_1) * beta_1 ** alpha_1 * np.exp( -beta_1 * t )) /gamma( alpha_1 )) - c *
            ( ( t ** (alpha_2) * beta_2 ** alpha_2 * np.exp( -beta_2 * t )) /gamma( alpha_2 )) )
            
    if integrator: # pragma: no cover
        hrf /= integrator(hrf)
        
    return hrf.astype(dtype)

def percent_change(ts, ax=-1):

    r"""Returns the % signal change of each point of the times series
    along a given axis of the array timeseries

    Parameters
    ----------
    ts : ndarray
        an array of time series

    ax : int, optional (default to -1)
        the axis of time_series along which to compute means and stdevs

    Returns
    -------
    ndarray
        the renormalized time series array (in units of %)

    Examples
    --------
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> ts = np.arange(4*5).reshape(4,5)
    >>> ax = 0
    >>> percent_change(ts,ax)
    array([[-100.    ,  -88.2353,  -78.9474,  -71.4286,  -65.2174],
           [ -33.3333,  -29.4118,  -26.3158,  -23.8095,  -21.7391],
           [  33.3333,   29.4118,   26.3158,   23.8095,   21.7391],
           [ 100.    ,   88.2353,   78.9474,   71.4286,   65.2174]])
    >>> ax = 1
    >>> percent_change(ts,ax)
    array([[-100.    ,  -50.    ,    0.    ,   50.    ,  100.    ],
           [ -28.5714,  -14.2857,    0.    ,   14.2857,   28.5714],
           [ -16.6667,   -8.3333,    0.    ,    8.3333,   16.6667],
           [ -11.7647,   -5.8824,    0.    ,    5.8824,   11.7647]])
    """
    ts = np.asarray(ts)

    return (ts / np.expand_dims(np.mean(ts, ax), ax) - 1) * 100


def detrend_psc(ts,ax=-1):
    ts_mean = np.mean(ts, axis=ax)[..., None]
    ts_detrend = detrend(ts, axis=ax, type='linear') + ts_mean
    ts_pct = percent_change(ts_detrend, ax=-1)
    return ts_pct


def zscore(time_series, axis=-1):

    r"""Returns the z-score of each point of the time series
    along a given axis of the array time_series.

    Parameters
    ----------
    time_series : ndarray
        an array of time series
    axis : int, optional
        the axis of time_series along which to compute means and stdevs

    Returns
    _______
    zt : ndarray
        the renormalized time series array
    """

    time_series = np.asarray(time_series)
    et = time_series.mean(axis=axis)
    st = time_series.std(axis=axis)
    sl = [slice(None)] * len(time_series.shape)
    sl[axis] = np.newaxis
    if sl == [None]:
        zt = (time_series - et)/st
    else:
        zt = time_series - et[sl]
        zt /= st[sl]
        
    return zt
    
def bootstrap_bundle(bootstraps, resamples, Fit, model, data, grids, bounds, indices, auto_fit=True, verbose=1, Ns=None):
    
    # initialze
    Fits = []
    
    # main loop
    for resample in resamples:
        for bootstrap in xrange(bootstraps):
            for voxel in xrange(data.shape[0]):
                
                # voxel
                voxel_idx = indices[voxel]
                
                # create random draws
                resample_idx = np.random.choice(np.arange(data.shape[1]-1),resample,replace=False)
                
                # data
                this_data = np.mean(data[voxel,resample_idx,:],0)
                
                # store it
                Fits.append((Fit, model, this_data, grids, bounds, Ns, voxel_idx, auto_fit, verbose, resample_idx))
                
    # randomize list order
    idx = np.argsort(np.random.rand(len(Fits)))
    Fits = [Fits[i] for i in idx]
    
    return Fits

def xval_bundle(bootstraps, kfolds, Fit, model, data, grids, bounds, indices, auto_fit=True, verbose=1, Ns=None):
    
    # num runs
    runs = np.arange(data.shape[1])
    
    # initialize
    Fits = []
    
    # main loop
    for bootstrap in xrange(bootstraps):
        for voxel in xrange(data.shape[0]):
            
            # voxel
            voxel_idx = indices[voxel]
            
            # data
            the_data = data[voxel,:,:]
            
            # create random draws
            if kfolds == 1: # leave one out
                trn_idx = np.random.choice(runs, len(runs)-1, replace=False)
            else:
                trn_idx = np.random.choice(runs, int(len(runs)/kfolds), replace=False)
                # trn_idx = np.random.choice(runs, np.int(len(runs)/kfolds), replace=False)
            
            tst_idx = np.array(list(set(runs)-set(trn_idx)))
            
            # compute mean timeseries
            trn_data = np.mean(the_data[trn_idx,:], 0)
            tst_data = np.mean(the_data[tst_idx,:], 0)
            
            # store it
            Fits.append((Fit, model, trn_data, tst_data, grids, bounds, Ns, voxel_idx, auto_fit, verbose, trn_idx, tst_idx))
    
    # randomize list order
    idx = np.argsort(np.random.rand(len(Fits)))
    Fits = [Fits[i] for i in idx]
    
    return Fits

def multiprocess_bundle(Fit, model, data, grids, bounds, indices, auto_fit=True, verbose=1, Ns=None):
    
    # num voxels
    num_voxels = int(np.shape(data)[0])
    
    # expand out grids and bounds
    grids = [grids,]*num_voxels
    bounds = [bounds,]*num_voxels
    
    # package the data structure
    dat = zip(repeat(Fit,num_voxels),
              repeat(model,num_voxels),
              data,
              grids,
              bounds,
              repeat(Ns,num_voxels),
              indices,
              repeat(auto_fit,num_voxels),
              repeat(verbose,num_voxels))
    
    # randomize list order
    dat = list(dat)
    idx = np.argsort(np.random.rand(len(dat)))
    dat = [dat[i] for i in idx]
    
    return dat

def gaussian_2D(X, Y, x0, y0, sigma_x, sigma_y, degrees, amplitude=1):
    
    theta = degrees*np.pi/180
    
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    
    Z = amplitude*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    
    return Z

def parallel_xval(args):

    r"""
    This is a convenience function for parallelizing the fitting
    procedure.  Each call is handed a tuple or list containing
    all the necessary inputs for instantiaing a `GaussianFit`
    class object and estimating the model parameters.
    
    
    Paramaters
    ----------
    args : list/tuple
        A list or tuple containing all the necessary inputs for fitting
        the Gaussian pRF model.
        
    Returns
    -------
    
    fit : `Fit` class object
        A fit object that contains all the inputs and outputs of the
        pRF model estimation for a single voxel.
        
    """
    
    # unpackage the arguments
    Fit = args[0]
    model = args[1]
    trn_data = args[2]
    tst_data = args[3]
    grids = args[4]
    bounds = args[5]
    voxel_index = args[6]
    Ns = args[7]
    auto_fit = args[8]
    verbose = args[9]
    trn_idx = args[10]
    tst_idx = args[11]
    
    # fit the data
    fit = Fit(model,
              trn_data,
              grids,
              bounds,
              Ns,
              voxel_index,
              auto_fit,
              verbose)
    
    fit.trn_data = trn_data
    fit.tst_data = tst_data
    fit.trn_idx = trn_idx
    fit.tst_idx = tst_idx
    fit.cod = coeff_of_determination(fit.tst_data, fit.prediction)
    
    return fit
        
def parallel_bootstrap(args):
    
    r"""
    This is a convenience function for parallelizing the fitting
    procedure.  Each call is handed a tuple or list containing
    all the necessary inputs for instantiaing a `GaussianFit`
    class object and estimating the model parameters.
    
    
    Paramaters
    ----------
    args : list/tuple
        A list or tuple containing all the necessary inputs for fitting
        the Gaussian pRF model.
        
    Returns
    -------
    
    fit : `Fit` class object
        A fit object that contains all the inputs and outputs of the
        pRF model estimation for a single voxel.
        
    """
    
    
    # unpackage the arguments
    Fit = args[0]
    model = args[1]
    data = args[2]
    grids = args[3]
    bounds = args[4]
    voxel_index = args[5]
    Ns = args[6]
    auto_fit = args[7]
    verbose = args[8]
    resamples = args[9]
    
    # fit the data
    fit = Fit(model,
              data,
              grids,
              bounds,
              Ns,
              voxel_index,
              auto_fit,
              verbose)
    
    fit.resamples = resamples
    fit.n_resamples = len(resamples)
    
    return fit


def parallel_fit(args):

    r"""
    This is a convenience function for parallelizing the fitting
    procedure.  Each call is handed a tuple or list containing
    all the necessary inputs for instantiaing a `GaussianFit`
    class object and estimating the model parameters.
    
    Paramaters
    ----------
    args : list/tuple
        A list or tuple containing all the necessary inputs for fitting
        the Gaussian pRF model.
        
    Returns
    -------
    
    fit : `Fit` class object
        A fit object that contains all the inputs and outputs of the
        pRF model estimation for a single voxel.
        
    """
    
    
    # unpackage the arguments
    Fit = args[0]
    model = args[1]
    data = args[2]
    grids = args[3]
    bounds = args[4]
    voxel_index = args[5]
    Ns = args[6]
    auto_fit = args[7]
    verbose = args[8]
    
    if True:
        # fit the data
        fit = Fit(model,
                  data,
                  grids,
                  bounds,
                  Ns,
                  voxel_index,
                  auto_fit,
                  verbose)
        return fit
    
    # except:

    #     fit = Fit(model,
    #               data,
    #               grids,
    #               bounds,
    #               Ns,
    #               voxel_index,
    #               False,
    #               verbose)
    #     return fit




def cartes_to_polar(cartes):

    """
    Assumes that the 0th and 1st parameters are cartesian `x` and `y`
    """
    polar = cartes.copy()
    polar[...,0] = np.mod(np.arctan2(cartes[...,1], cartes[...,0]),2*np.pi)
    polar[...,1] = np.sqrt(cartes[...,0]**2 + cartes[...,1]**2)
    return polar

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

# def generate_polar_grid(screen_dva,N,sigma_min=0.1,n_min=0.1,n_max=1.25):
#     """
    

#     Parameters
#     ----------
#     bounds : array
#         Bounds on rf parameters [(X),(Y),(sigma),(n)].

#     Returns
#     -------
#     Array of gridpoints.

#     """
    
#     #first generate x,y grids
#     xy_basegrid = np.exp(np.linspace(0,np.log(screen_dva),N//2))
#     xy_grid = np.concatenate((np.append(-np.flip(xy_basegrid),0.0),xy_basegrid))
    
#     Nnew = len(xy_grid)
    
#     sigma_grid = np.exp(np.linspace(np.log(sigma_min),np.log(screen_dva/2),Nnew))

    
    


def find_files(directory, pattern):
    names = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                names.append(filename)
    return names

def binner(signal, times, bins):
    binned_response = np.zeros(len(bins)-2)
    bin_width = bins[1] - bins[0]
    for t in xrange(1,len(bins)):
        the_bin = bins[t]
        binned_signal = signal[(times >= the_bin-bin_width) & (times <= the_bin)]
        binned_response[t-2] = np.sum(binned_signal)
    return binned_response

import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    #PEAKDET Detect peaks in a vector
    #        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    #        maxima and minima ("peaks") in the vector V.
    #        MAXTAB and MINTAB consists of two columns. Column 1
    #        contains indices in V, and column 2 the found values.
    #
    #        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    #        in MAXTAB and MINTAB are replaced with the corresponding
    #        X-values.
    #
    #        A point is considered a maximum peak if it has the maximal
    #        value, and was preceded (to the left) by a value lower by
    #        DELTA.
    #
    # Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    # This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
    
    if x is None: # pragma: no cover
        x = arange(len(v))
        
    v = asarray(v)
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
            
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

def coeff_of_determination(data, model, axis=-1):
    
    r"""
    Calculate the coefficient of determination for a model prediction, relative
    to data.
    
    Parameters
    ----------
    data : ndarray
        The data
    model : ndarray
        The predictions of a model for this data. Same shape as the data.
    axis: int, optional
        The axis along which different samples are laid out (default: -1).
        
    Returns
    -------
    COD : ndarray
       The coefficient of determination.
       
    """
    
    residuals = data - model
    ss_err = np.sum(residuals ** 2, axis=axis)
    
    demeaned_data = data - np.mean(data, axis=axis)[..., np.newaxis]
    ss_tot = np.sum(demeaned_data **2, axis=axis)
    
    # Don't divide by 0:
    if np.all(ss_tot==0.0):
        return np.nan
            
    return 100 * (1 - (ss_err/ss_tot))

