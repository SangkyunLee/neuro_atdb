#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:30:55 2020
many functions and classes are adapted from tolias lab codes.
@author: Sang
"""


from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import numpy as np

from functools import partial
import datajoint as dj

class NaNSpline(InterpolatedUnivariateSpline):
    def __init__(self, x, y, **kwargs):
        xnan = np.isnan(x)
        if np.any(xnan):
            print('Found nans in the x-values. Replacing them with linear interpolation')
        ynan = np.isnan(y)
        w = xnan | ynan  # get nans
        x, y = map(np.array, [x, y])  # copy arrays
        y[ynan] = 0
        x[xnan] = np.interp(np.where(xnan)[0], np.where(~xnan)[0], x[~xnan])
        super().__init__(x[~w], y[~w], **kwargs)  # assign zero weight to nan positions

        self.nans = interp1d(x, 1 * w, kind='linear')

    def __call__(self, x, **kwargs):
        ret = np.zeros_like(x)
        newnan = np.zeros_like(x)

        old_nans = np.isnan(x)
        newnan[old_nans] = 1
        newnan[~old_nans] = self.nans(x[~old_nans])

        idx = newnan > 0
        ret[idx] = np.nan
        ret[~idx] = super().__call__(x[~idx], **kwargs)
        return ret
    
def fill_nans(x, preserve_gap=None):
    """
    :param x:  1D array  -- will
    :return: the array with nans interpolated
    The input argument is modified.
    """
    if preserve_gap is not None:
        assert preserve_gap % 2 == 1, 'can only efficiently preserve odd gaps'
        keep = np.convolve(np.convolve(1 * np.isnan(x), np.ones(preserve_gap), mode='same') == preserve_gap,
                           np.ones(preserve_gap, dtype=bool), mode='same')
    else:
        keep = np.zeros(len(x), dtype=bool)

    nans = np.isnan(x)

    x[nans] = 0 if nans.all() else np.interp(nans.nonzero()[0], (~nans).nonzero()[0], x[~nans])
    x[keep] = np.nan
    return x

class RangeError(Exception):
    def __init__(self, message):
        self.message =message
        

def get_filter(dur_sec, sampling_timesec, type='hamming'):
    M = int(dur_sec/sampling_timesec)
    if M<=0:
        raise RangeError("dur_sec:{} should be greater than sampling_timesec:{}".format(dur_sec,sampling_timesec))
    else:
        if type == 'hamming':
            h = np.hamming(2*M+1)
            h = h/np.sum(h)
        else:
            raise NotImplementedError('Filter {}  not implemented'.format(filter))
    return h
                
            


#def load_eye_traces(key):
#    
#    pupil = dj.create_virtual_module('pupil', 'pipeline_eye')
#    
#    # FittedPupil.Circle has center y are missing in some scans
#    #r, center = (pupil.FittedPupil.Circle() & key).fetch('radius', 'center', order_by='frame_id')
#    
#    tracking_method = (dj.U('tracking_method')&(pupil.FittedPupil.Ellipse & key)).fetch('tracking_method')
#
#    if len(tracking_method)==0:        
#        return [],[],[]
#    
#    r, center = (pupil.FittedPupil.Ellipse & 
#                 {**key, 'tracking_method': tracking_method[-1]}).fetch(
#                         'major_radius', 'center', order_by='frame_id')
# 
#    detectedFrames = ~np.isnan(r)
#    xy = np.full((len(r), 2), np.nan)
#    xy[detectedFrames, :] = np.vstack(center[detectedFrames])
#    xy = np.vstack(map(partial(fill_nans, preserve_gap=3), xy.T))
#    if np.any(np.isnan(xy)):
#        print(key,end=':')
#        print(' Keeping some nans in the pupil location trace')
#    pupil_radius = fill_nans(r.squeeze(), preserve_gap=3)
#    if np.any(np.isnan(pupil_radius)):
#        print(key,end=':')
#        print('Keeping some nans in the pupil radius trace')
#
#    eye_time = (pupil.Eye() & key).fetch1('eye_time').squeeze()
#    return pupil_radius, xy, eye_time


def load_eye_traces(key, shape='circle', method =None):
    
    pupil = dj.create_virtual_module('pupil', 'pipeline_eye')
    
    # FittedPupil.Circle has center y are missing in some scans
    #r, center = (pupil.FittedPupil.Circle() & key).fetch('radius', 'center', order_by='frame_id')
    if shape =='circle':
        track_shape = pupil.FittedPupil.Circle & key
    else:
        track_shape = pupil.FittedPupil.Ellipse & key
    if method:
        tracking_method = method
    else:        
        tracking_method = (dj.U('tracking_method')& track_shape).fetch('tracking_method')
        if len(tracking_method)>0:
            tracking_method = tracking_method[-1]
        
    
    if not tracking_method:    # tracking method is not specified    
        return [],[],[], None
        
    if shape =='circle':
        r, center = (track_shape & {'tracking_method': tracking_method}).fetch(    
                'radius', 'center', order_by='frame_id')
    else:
        r, center = (track_shape & {'tracking_method': tracking_method}).fetch(                         
                'major_radius', 'center', order_by='frame_id')
        
   
    detectedFrames = ~np.isnan(r)
    xy = np.full((len(r), 2), np.nan)
    xy[detectedFrames, :] = np.vstack(center[detectedFrames])
    xy_ = list(map(partial(fill_nans, preserve_gap=3), xy.T))
    xy = np.vstack(xy_)
    if np.any(np.isnan(xy)):
        print(key,end=':')
        print(' Keeping some nans in the pupil location trace')
    pupil_radius = fill_nans(r.squeeze(), preserve_gap=3)
    if np.any(np.isnan(pupil_radius)):
        print(key,end=':')
        print('Keeping some nans in the pupil radius trace')

    eye_time = (pupil.Eye() & key).fetch1('eye_time').squeeze()
    return pupil_radius, xy, eye_time, tracking_method








def load_frame_times_vstim(key):
    """
    load 2p frame times on visual stimulus clock
    """
    k = dict(key)
    k.pop('field', None)
    stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
    meso = dj.create_virtual_module('meso', 'pipeline_meso')
    
    ndepth = len(dj.U('z') & (meso.ScanInfo.Field() & k))
    return (stimulus.Sync() & key).fetch1('frame_times').squeeze()[::ndepth]


def load_frame_times_behav(key):  
    """
    load 2p frame times on behav clock
    """
    k = dict(key)
    k.pop('field', None)
    stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
    meso = dj.create_virtual_module('meso', 'pipeline_meso')
    
    ndepth = len(dj.U('z') & (meso.ScanInfo.Field() & k))
    return (stimulus.BehaviorSync() & key).fetch1('frame_times').squeeze()[0::ndepth]

def load_treadmill_trace(key):
    treadmill = dj.create_virtual_module('treadmill', 'pipeline_treadmill')
    t, v = (treadmill.Treadmill() & key).fetch1('treadmill_time', 'treadmill_vel')
    v = v.squeeze()
    if np.any(np.isnan(v)):
        print(key,end=':')
        print(' Keeping some nans in the treadmil trace')    
    return v, t.squeeze()
