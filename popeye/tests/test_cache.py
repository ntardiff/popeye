import os
import sharedmem
import ctypes
import pickle
import time

import numpy as np
import numpy.testing as npt
import nose.tools as nt

import popeye.utilities as utils
import popeye.og as og
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus

def test_cache_model():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,90)
    thetas = np.insert(thetas,0,-1)
    thetas = np.append(thetas,-1)
    num_blank_steps = 20
    num_bar_steps = 20
    ecc = 10
    tr_length = 1.5
    frames_per_tr = 1.0
    scale_factor = 0.50
    pixels_across = 100
    pixels_down = 100
    dtype = ctypes.c_int16
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, 
                                thetas, num_bar_steps, num_blank_steps, ecc, clip=0.01)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    model.hrf_delay = 0
    model.mask_size = 5
    
    # set cache grids
    x_grid = utils.grid_slice(-10, 10, 10)
    y_grid = utils.grid_slice(-10, 10, 10)
    s_grid = utils.grid_slice(0.55,5.25, 10)
    grids = (x_grid, y_grid, s_grid,)
    
    # seed rng
    np.random.seed(4932)
    
    # cache the pRF model
    cache = model.cache_model(grids, ncpus=sharedmem.cpu_count()-1);
    
    # save it out
    pickle.dump(cache, open('/tmp/og_cached_model.pkl','wb'))
    
    # make sure its the right size
    cached_model = pickle.load(open('/tmp/og_cached_model.pkl','rb'))
    
    nt.assert_equal(np.sum([c[0] for c in cache]),np.sum([c[0] for c in cached_model]))

def test_resurrect_model():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,90)
    thetas = np.insert(thetas,0,-1)
    thetas = np.append(thetas,-1)
    num_blank_steps = 20
    num_bar_steps = 20
    ecc = 10
    tr_length = 1.5
    frames_per_tr = 1.0
    scale_factor = 0.50
    pixels_across = 100
    pixels_down = 100
    dtype = ctypes.c_int16
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, 
                                thetas, num_bar_steps, num_blank_steps, ecc, clip=0.01)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # set cache grids
    x_grid = utils.grid_slice(-10, 10, 10)
    y_grid = utils.grid_slice(-10, 10, 10)
    s_grid = utils.grid_slice(0.55,5.25, 10)
    grids = (x_grid, y_grid, s_grid,)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    model.hrf_delay = 0
    model.mask_size = 5
    
    # seed rng
    np.random.seed(4932)
    
    # cache the model
    cache = model.cache_model(grids, ncpus=sharedmem.cpu_count()-1);
    
    # resurrect cached model
    cached_model_path = '/tmp/og_cached_model.pkl'
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf, cached_model_path=cached_model_path)
    model.hrf_delay = 0
    model.mask_size = 5
    
    # make sure the same
    nt.assert_true(np.sum([c[0] for c in cache] -  model.cached_model_timeseries) == 0)
    nt.assert_true(np.sum([c[1] for c in cache] -  model.cached_model_parameters) == 0)
    