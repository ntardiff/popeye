import numpy as np
import socket, os, shutil
from scipy.signal import detrend
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error

hostname = socket.gethostname()

def set_paths(params):
    subjID = params['subjID']
    p = {}
    if hostname == 'syndrome' or hostname == 'zod.psych.nyu.edu' or hostname == 'zod':
        # If one of the lab computers with local mount of data server
        p['pRF_data'] = '/d/DATA/data/popeye_pRF/'
        p['orig_data'] = '/d/DATD/datd/pRF_orig/'
    else: # Set paths on local macbook of Mrugank
        p['pRF_data'] = '/Users/mrugankdake/Documents/Clayspace/MRI/popeye_pRF/'
        p['orig_data'] = '/Users/mrugankdake/Documents/Clayspace/MRI/pRF_orig/'
    # else:
        # Set paths on HPC
    p['stimuli_path'] = os.path.join(p['pRF_data'], 'Stimuli')
    p['gridfit_path'] = os.path.join(p['stimuli_path'], 'gridfit.npy')
    # Paths for relevant files from the original data
    p['orig_brainmask'] = os.path.join(p['orig_data'], subjID, 'surfanat_brainmask_hires.nii.gz')
    p['orig_func'] = os.path.join(p['orig_data'], subjID, 'RF1', subjID+'_RF1_vista', 'bar_seq_1_func.nii.gz')
    p['orig_ss5'] = os.path.join(p['orig_data'], subjID, 'RF1', subjID+'_RF1_vista', 'bar_seq_1_ss5.nii.gz')
    p['orig_surf'] = os.path.join(p['orig_data'], subjID, 'RF1', subjID+'_RF1_vista', 'bar_seq_1_surf.nii.gz')
    p['orig_anat'] = os.path.join(p['orig_data'], subjID, 'anat_T1_brain.nii')

    # Paths for the new pRF holder
    p['pRF_brainmask'] = os.path.join(p['pRF_data'], subjID, 'surfanat_brainmask_hires.nii.gz')
    p['pRF_func'] = os.path.join(p['pRF_data'], subjID, 'bar_seq_1_func.nii.gz')
    p['pRF_ss5'] = os.path.join(p['pRF_data'], subjID, 'bar_seq_1_ss5.nii.gz')
    p['pRF_surf'] = os.path.join(p['pRF_data'], subjID, 'bar_seq_1_surf.nii.gz')
    p['pRF_anat'] = os.path.join(p['pRF_data'], subjID, 'anat_T1_brain.nii')

    # Figure directory
    p['fig_dir'] = os.path.join(p['pRF_data'], subjID, 'figures')
    if not os.path.exists(p['fig_dir']):
        os.mkdir(p['fig_dir'])

    # Copy a folder as a hyperlink
    if hostname == 'syndrome' or hostname == 'zod.psych.nyu.edu' or hostname == 'zod':
        p['orig_anat_dir'] = os.path.join(p['orig_data'], subjID, subjID+'anat')
        p['pRF_anat_dir'] = os.path.join(p['pRF_data'], subjID, subjID+'anat')
        if not os.path.exists(p['pRF_anat_dir']):
            shutil.copy(p['orig_anat_dir'], os.path.join(p['pRF_data'], subjID), follow_symlinks=False)
    return p

def load_stimuli(p):
    '''
    Loads the bar and the params files that should be constant across subjects.
    '''
    bar = loadmat(os.path.join(p['stimuli_path'], 'bar_stimulus_masks_1300ms_images.mat'))['images']
    params = loadmat(os.path.join(p['stimuli_path'], 'bar_stimulus_masks_1300ms_params.mat'))
    return bar, params


def copy_files(p, params):
    subjID = params['subjID']
    if not os.path.exists(os.path.join(p['pRF_data'], subjID)):
        os.mkdir(os.path.join(p['pRF_data'], subjID))
        os.system('cp ' + p['orig_brainmask'] + ' ' + p['pRF_brainmask'])
        os.system('cp ' + p['orig_func'] + ' ' + p['pRF_func'])
        os.system('cp ' + p['orig_ss5'] + ' ' + p['pRF_ss5'])
        os.system('cp ' + p['orig_surf'] + ' ' + p['pRF_surf'])
        os.system('cp ' + p['orig_anat'] + ' ' + p['pRF_anat'])
    else:
        print('Subject folder already exists')

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

def print_time(st_time, end_time, process_name):
    duration = end_time - st_time
    if duration < 60:
        print(f'{process_name} took {round(duration, 2)} seconds')
    elif duration < 3600:
        print(f'{process_name} took {round(duration/60, 2)} minutes ({round(duration//60, 2)} seconds)')

def constraint_grids(grid_space_orig, stimulus):
    print(f'Number of grid points: {len(grid_space_orig)}')
    idxs_to_drop = []
    for i in range(len(grid_space_orig)):
        if np.sqrt(grid_space_orig[i][0]**2 + grid_space_orig[i][1]**2) >= 2*stimulus.deg_x0.max():
            idxs_to_drop.append(i)
        if np.sqrt(grid_space_orig[i][0]**2 + grid_space_orig[i][1]**2) >= stimulus.deg_x0.max() + 2*grid_space_orig[i][2]:
            idxs_to_drop.append(i)
    grid_space = [grid_space_orig[i] for i in range(len(grid_space_orig)) if i not in idxs_to_drop]
    print(f'Number of grid points after dropping: {len(grid_space)}')
    return grid_space

def generate_bounds(init_estim, param_width):
    x_estim, y_estim, sigma_estim, n_estim = init_estim[5], init_estim[6], init_estim[3], init_estim[4]
    
    [x_bound_min, x_bound_max] = [x_estim - param_width[0], x_estim + param_width[0]]
    [y_bound_min, y_bound_max] = [y_estim - param_width[1], y_estim + param_width[1]]
    [sigma_bound_min, sigma_bound_max] = [sigma_estim - param_width[2], sigma_estim + param_width[2]]
    [n_bound_min, n_bound_max] = [n_estim - param_width[3], n_estim + param_width[3]]
    x_bounds = (x_bound_min, x_bound_max)
    y_bounds = (y_bound_min, y_bound_max)
    sigma_bounds = (sigma_bound_min, sigma_bound_max)
    n_bounds = (n_bound_min, n_bound_max)    
    bounds = (x_bounds, y_bounds, sigma_bounds, n_bounds)
    return bounds

def error_func(parameters, data, stimulus, objective_function):
    prediction = objective_function(*parameters, stimulus)
    error = mean_squared_error(data, prediction, squared=True)
    return error