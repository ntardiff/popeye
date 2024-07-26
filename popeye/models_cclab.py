#!/usr/bin/python

""" Classes and functions for estimating Gaussian pRF model """

from __future__ import division
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import linregress,zscore
#import nibabel

from popeye.onetime import auto_attr
import popeye.utilities_cclab as utils
#from popeye.utilities_cclab import generate_og_receptive_field,generate_rf_timeseries
from popeye.base_cclab import PopulationModel, PopulationFit
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries, generate_rf_timeseries_nomask
#from popeye import spinach

class GaussianModel(PopulationModel):
    
    def __init__(self, *args, **kwargs): #stimulus, hrf_model, normalizer=utils.percent_change, cached_model_path=None, nuisance=None):
        
        r"""A 2D Gaussian population receptive field model [1]_.
        
        Paramaters
        ----------
        
        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.
            
        hrf_model : callable
            A function that generates an HRF model given an HRF delay.
            For more information, see `popeye.utilties.double_gamma_hrf_hrf`
        
        
        References
        ----------
        
        .. [1] Dumoulin SO, Wandell BA. (2008) Population receptive field 
        estimates in human visual cortex. NeuroImage 39:647-660
        
        """
        
        PopulationModel.__init__(self, *args, **kwargs) #stimulus, hrf_model, normalizer)
        
    # main method for deriving model time-series
    def generate_ballpark_prediction(self, x, y, sigma):
        
        r"""
        Predict signal for the Gaussian Model using the downsampled stimulus.
        The rate of stimulus downsampling is defined in `model.stimulus.scale_factor`.
        
        Parameters
        __________
        x : float
            Horizontal location of the Gaussian RF.
        
        y: float 
            Vertical location of the Gaussian RF.
        
        sigma: float
            Dipsersion of the Gaussian RF.
        
        """
        
        # mask for speed
        mask = self.distance_mask_coarse(x, y, sigma)
        
        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x0, self.stimulus.deg_y0)
        rf /= (2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x0[0,0:2])**2
                
        # extract the stimulus time-series
        response = generate_rf_timeseries(self.stimulus.stim_arr0, rf)
        
        # convolve it with the stimulus
        model = fftconvolve(response, self.hrf())[0:len(response)]
        
        # units
        model = self.normalizer(model)
        
        # regress out mean and amplitude
        beta, baseline = self.regress(model, self.data)
        
        # scale
        model *= beta
        
        # offset
        model += baseline
        
        return model
        
    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, beta, baseline, unscaled=False):
        
        r"""
        Predict signal for the Gaussian Model.
        
        Parameters
        __________
        x : float
            Horizontal location of the Gaussian RF.
        
        y: float 
            Vertical location of the Gaussian RF.
        
        sigma: float
            Dipsersion of the Gaussian RF.
        
        beta : float
            Amplitude scaling factor to account for units.
        
        baseline: float
            Amplitude intercept to account for baseline.
        
        """
        
        # mask for speed
        mask = self.distance_mask(x, y, sigma)
        
        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        rf /= (2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x[0,0:2])**2
        
        # extract the stimulus time-series
        response = generate_rf_timeseries(self.stimulus.stim_arr, rf)
        
        # convolve it with the stimulus
        model = fftconvolve(response, self.hrf())[0:len(response)]
        
        # units
        model = self.normalizer(model)
        
        if unscaled:
            return model
        else:
            
            # scale it by beta
            model *= beta
            
            # offset
            model += baseline
            
            return model
    
    def generate_receptive_field(self, x, y, sigma):
        
        r"""
        Generate a Gaussian receptive field in stimulus-referred coordinates.
        
        Parameters
        __________
        x : float
            Horizontal location of the Gaussian RF.
            
        y: float 
            Vertical location of the Gaussian RF.
            
        sigma: float
            Dipsersion of the Gaussian RF.
            
        """
        
        rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        rf /= (2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x[0,0:2])**2
        
        return rf
        
class GaussianFit(PopulationFit):
    
    r"""
    A 2D Gaussian population receptive field fit class.
    
    """
    
    def __init__(self, *args, **kwargs): #model, data, grids, bounds,
                 #voxel_index=(1,2,3), Ns=None, auto_fit=True, verbose=0):
        
        r"""
        A class containing tools for fitting the 2D Gaussian pRF model.
        
        The `GaussianFit` class houses all the fitting tool that are associated with 
        estimatinga pRF model.  The `GaussianFit` takes a `GaussianModel` instance 
        `model` and a time-series `data`.  In addition, extent and sampling-rate of a 
        brute-force grid-search is set with `grids` and `Ns`.  Use `bounds` to set 
        limits on the search space for each parameter.  
        
        Paramaters
        ----------
        
        
        model : `GaussianModel` class instance
            An object representing the 2D Gaussian model. 
            
        data : ndarray
            An array containing the measured BOLD signal of a single voxel.
            
        grids : tuple or Slice Object
            A tuple indicating the search space for the brute-force grid-search.
            The tuple contains pairs of upper and lower bounds for exploring a
            given dimension.  For example `grids=((-10,10),(0,5),)` will
            search the first dimension from -10 to 10 and the second from 0 to 5.
            The resolution of this search space is set with the `Ns` argument. 
            For more information, see `scipy.optimize.brute`.
            
            Alternatively you can pass `grids` a Slice Object. If you do use this
            option, you do not need to specificy `Ns` in `GaussianFit`. See 
            `popeye.utilities.grid_slice` for more details.
            
        bounds : tuple
            A tuple containing the upper and lower bounds for each parameter
            in `parameters`.  If a parameter is not bounded, simply use
            `None`.  For example, `fit_bounds=((0,None),(-10,10),)` would 
            bound the first parameter to be any positive number while the
            second parameter would be bounded between -10 and 10.
            
        voxel_index : tuple
            A tuple containing the index of the voxel being modeled. The 
            fitting procedure does not require a voxel index, but 
            collating the results across many voxels will does require voxel
            indices. With voxel indices, the brain volume can be reconstructed 
            using the newly computed model estimates.
            
        Ns : int
            Number of samples per stimulus dimension to sample during the ballpark search.
            For more information, see `scipy.optimize.brute`.
            
            This can be `None` if `grids` is a tuple of Slice Objects.
            
        auto_fit : bool
            A flag for automatically running the fitting procedures once the 
            `GaussianFit` object is instantiated.
            
        verbose : int
            0 = silent
            1 = print the final solution of an error-minimization
            2 = print each error-minimization step
            
        """
        
        PopulationFit.__init__(self, *args, **kwargs) #model, data, grids, bounds, 
                           #voxel_index, Ns, auto_fit, verbose)
                           
    @auto_attr
    def overloaded_estimate(self):
        return [self.theta, self.rho, self.sigma, self.beta, self.baseline]
    
    @auto_attr
    def x0(self):
        return self.ballpark[0]
        
    @auto_attr
    def y0(self):
        return self.ballpark[1]
        
    @auto_attr
    def s0(self):
        return self.ballpark[2]
    
    @auto_attr
    def beta0(self):
        return self.ballpark[3]
        
    @auto_attr
    def baseline0(self):
        return self.ballpark[4]
            
    @auto_attr
    def x(self):
        return self.estimate[0]
        
    @auto_attr
    def y(self):
        return self.estimate[1]
        
    @auto_attr
    def sigma(self):
        return self.estimate[2]
    
    @auto_attr
    def beta(self):
        return self.estimate[3]
    
    @auto_attr
    def baseline(self):
        return self.estimate[4]
        
    @auto_attr
    def rho(self):
        
        r""" Returns the eccentricity of the fitted pRF. """
        
        return np.sqrt(self.x**2+self.y**2)
    
    @auto_attr
    def theta(self):
        
        r""" Returns the polar angle of the fitted pRF. """
        
        return np.mod(np.arctan2(self.y,self.x),2*np.pi)
    
    @auto_attr
    def receptive_field(self):
        
        r""" Returns the fitted Gaussian pRF. """
        
        return self.model.generate_receptive_field(self.x, self.y, self.sigma)
                                           


class CompressiveSpatialSummationModel(PopulationModel):
    
    r"""
    A Compressive Spatial Summation population receptive field model class
    
    """
    
    def __init__(self, *args, **kwargs): #stimulus, hrf_model, normalizer=utils.percent_change, cached_model_path=None, 
                 #nuisance=None):
        
        r"""
        A Compressive Spatial Summation population receptive field model [1]_.
        
        Paramaters
        ----------
        
        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.
        
        hrf_model : callable
            A function that generates an HRF model given an HRF delay.
            For more information, see `popeye.utilties.double_gamma_hrf_hrf`
        
        References
        ----------
        
        .. [1] Kay KN, Winawer J, Mezer A, Wandell BA (2014) Compressive spatial
        summation in human visual cortex. Journal of Neurophysiology 110:481-494.
        
        """
        
        PopulationModel.__init__(self, *args, **kwargs) #stimulus, hrf_model, normalizer, cached_model_path, nuisance)
        
    # main method for deriving model time-series
    def generate_ballpark_prediction(self, x, y, sigma, n, unscaled=False):
        
        # mask = self.distance_mask_coarse(x, y, sigma)

        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma,self.stimulus.deg_x0, self.stimulus.deg_y0)
        
        # normalize by the integral (this is not necessary if normalizing below)
        #rf /= ((2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x0[0,0:2])**2)
        
        # extract the stimulus time-series
        #response = generate_rf_timeseries(self.stimulus.stim_arr0, rf, mask)
        #response = generate_rf_timeseries(self.stimulus.stim_arr0, rf)
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr0, rf)
        
        # compression
        response **= n
        
        # convolve with the HRF
        # hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        # # convolve it with the stimulus
        # model = fftconvolve(response, hrf)[0:len(response)]
        model = fftconvolve(response, self.hrf())[0:len(response)]
        
        # units
        model = self.normalizer(model)

        # units
        #model = zscore(model) #(model - np.mean(model)) / np.mean(model)
        
        # regress out mean and linear
        p = linregress(model, self.data)
        
        # scale
        model *= p[0]
        
        # offset
        model += p[1]
        
        return model

        
    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, n, beta, baseline, unscaled=False):
        
        # mask = self.distance_mask(x, y, sigma)

        # generate the RF
        ###rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        
        # normalize by the integral (this is not necessary if normalizing below)
        #rf /= ((2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x[0,0:2])**2)
        
        # extract the stimulus time-series
        #response = generate_rf_timeseries(self.stimulus.stim_arr, rf, mask)
        #response = generate_rf_timeseries(self.stimulus.stim_arr, rf)
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr, rf)
        
        # compression
        response **= n
        
        # convolve with the HRF
        # hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # # convolve it with the stimulus
        # model = fftconvolve(response, hrf)[0:len(response)]
        model = fftconvolve(response, self.hrf())[0:len(response)]
        
        # units
        model = self.normalizer(model)
        
        # convert units
        #model = zscore(model) #(model - np.mean(model)) / np.mean(model)
        
        if unscaled:
            return model
        else:
            
            # scale it by beta
            model *= beta
            
            # offset
            model += baseline
            
            return model
        
class CompressiveSpatialSummationFit(PopulationFit):
    
    """
    A Compressive Spatial Summation population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds=None, *args, **kwargs):
                 #voxel_index=(1,2,3), Ns=None, auto_fit=True, grid_only=False, verbose=0):
        
        
        r"""
        A class containing tools for fitting the CSS pRF model.
        
        The `CompressiveSpatialSummationFit` class houses all the fitting tool that 
        are associated with estimating a pRF model. The `GaussianFit` takes a 
        `CompressiveSpatialSummationModel` instance  `model` and a time-series `data`. 
        In addition, extent and sampling-rate of a  brute-force grid-search is set 
        with `grids` and `Ns`.  Use `bounds` to set limits on the search space for 
        each parameter.  
        
        Paramaters
        ----------
        
                
        model : `CompressiveSpatialSummationModel` class instance
            An object representing the CSS model. 
        
        data : ndarray
            An array containing the measured BOLD signal of a single voxel.
        
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
        
        voxel_index : tuple
            A tuple containing the index of the voxel being modeled. The 
            fitting procedure does not require a voxel index, but 
            collating the results across many voxels will does require voxel
            indices. With voxel indices, the brain volume can be reconstructed 
            using the newly computed model estimates.
        
        auto_fit : bool
            A flag for automatically running the fitting procedures once the 
            `GaussianFit` object is instantiated.
        
        verbose : int
            0 = silent
            1 = print the final solution of an error-minimization
            2 = print each error-minimization step
        
        """
        
        #if user doesn't supply bounds for optimization, let's generate some that seem reasonable
        #probably should allow some of these values to be set as defaults at some piont...
        if bounds is None:
            x_bounds = (-model.stimulus.screen_dva, model.stimulus.screen_dva)
            y_bounds = (-model.stimulus.screen_dva, model.stimulus.screen_dva)
            s_bounds = (0.5/model.stimulus.ppd, model.stimulus.screen_dva/2)
            n_bounds = (0.01, 1.25)
            b_bounds = (1e-8, None)
            m_bounds = (None, None)
            bounds = (x_bounds, y_bounds, s_bounds, n_bounds, b_bounds, m_bounds)

        
        PopulationFit.__init__(self, model, data, grids, bounds, 
                               *args, **kwargs) #voxel_index, Ns, auto_fit, grid_only, verbose)
    
    @auto_attr
    def overloaded_estimate(self):
        return [self.theta, self.rho, self.sigma_size, self.n, self.beta, self.baseline]
            
    @auto_attr
    def x0(self):
        return self.ballpark[0]
        
    @auto_attr
    def y0(self):
        return self.ballpark[1]
        
    @auto_attr
    def s0(self):
        return self.ballpark[2]
        
    @auto_attr
    def n0(self):
        return self.ballpark[3]
        
    @auto_attr
    def beta0(self):
        return self.ballpark[4]
        
    @auto_attr
    def baseline0(self):
        return self.ballpark[5]
    
    @auto_attr
    def rho0(self):
        return np.sqrt(self.x0**2+self.y0**2) if self.fit_method!='global_opt' else None
    
    @auto_attr
    def theta0(self):
        return np.mod(np.arctan2(self.y0,self.x0),2*np.pi) if self.fit_method!='global_opt' else None 
    
    @auto_attr
    def x(self):
        if self.fit_method=='grid_only':
            return self.ballpark[0]
        else:
            return self.estimate[0]
        
    @auto_attr
    def y(self):
        if self.fit_method=='grid_only':
            return self.ballpark[1]
        else:
            return self.estimate[1]
        
    @auto_attr
    def sigma(self):
        if self.fit_method=='grid_only':
            return self.ballpark[2]
        else:
            return self.estimate[2]
        
    @auto_attr
    def n(self):
        if self.fit_method=='grid_only':
            return self.ballpark[3]
        else:
            return self.estimate[3]
        
    @auto_attr
    def beta(self):
        if self.fit_method=='grid_only':
            return self.ballpark[4]
        else:
            return self.estimate[4]
    
    @auto_attr
    def baseline(self):
        if self.fit_method=='grid_only':
            return self.ballpark[5]
        else:
            return self.estimate[5]
        
    @auto_attr
    def rho(self):
        return np.sqrt(self.x**2+self.y**2)
    
    @auto_attr
    def theta(self):
        return np.mod(np.arctan2(self.y,self.x),2*np.pi)
    
    @auto_attr
    def sigma_size(self):
        return self.sigma / np.sqrt(self.n)
    