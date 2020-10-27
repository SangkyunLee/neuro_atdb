#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:36:38 2020

@author: sang
"""

#
%run access_db.py
#runfile('access_db.py')

import datajoint as dj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipeline import experiment, reso, meso, fuse, stack,  treadmill, pupil, shared
from stimulus import stimulus
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')  


schema = dj.schema('sang_spon')
@schema
class Sponscan(dj.Computed):
    definition = """ # scanlist with spontaneous recording
    
    -> meso.ScanInfo
    ---
    spon_frame_start                  : int unsigned      #frame start index
    spon_frame_dur                    : int unsigned      #frame duration
    caclulation_time=CURRENT_TIMESTAMP    : timestamp     # automatic
    """
    
    
    def make(self, key):
        
        
        stim_key = stimulus.Sync & key
        flip_times = (stimulus.Trial & key).fetch('flip_times')
        if len(stim_key)>0 and len(flip_times)>0:
            
            last_flip_times = np.nanmax(flip_times[-1].flatten())
            
            frame_times = stim_key.fetch1('frame_times').squeeze() 
            slice_num = len(np.unique((meso.ScanInfo.Field & key).fetch('z')))
            field_offset=0
            if slice_num>0:
                frame_times = frame_times[field_offset::slice_num]
            
                
            print(key)
            spon_frames = np.where(frame_times>last_flip_times)
            if len(spon_frames)>0 and len(spon_frames[0])>0:
                
                # print(spon_frames[0].shape)
                spon_frame_start = spon_frames[0][0]
                spon_dur = len(frame_times)-spon_frame_start
                
                out_= key.copy()
                out_['spon_frame_start'] =spon_frame_start
                out_['spon_frame_dur'] = spon_dur
                
                
                print(out_)
    
                
                self.insert1(out_)
                
        
# Sponscan.populate()        

#animal_ids: 17797, 17977,18142, 18252

    
@schema
class SponScanSel(dj.Computed):
    definition = """ # scan depth summary
    
    -> Sponscan
    depth_thr           : float         # max depth threshold for scan selection
    depth_interval_thr  : float         # max depth interval threshold for scan selection
    spon_framedur_thr   : int unsigned  # threshold for spontaneous frame duration 
    ---
    spon_frame_dur                    : int unsigned      #frame duration
    ndepth                            : int unsigned      # number of field depths
    field_depth                       : blob              #field depth list
    caclulation_time=CURRENT_TIMESTAMP    : timestamp     # automatic
    """
    def make(self, key):
        import numpy as np
        key1 = key.copy()
        key1['depth_thr'] = 450
        key1['depth_interval_thr'] = 100
        key1['spon_framedur_thr'] = 3000
        
        
        # field depth, this is not an accurate depth for individual cells
        depths = (meso.ScanInfo.Field & key).fetch('z')
        maxz = np.nan
        max_depth_interval = np.nan
        
        
        if len(depths)>1:            
            print(key)
            maxz = depths.max()
            udepth = np.unique(depths)
            if len(udepth)>1:
                max_depth_interval = np.max(np.diff(udepth))
        
        spon_frame_dur = (Sponscan() &key).fetch1('spon_frame_dur')
        
        if maxz > key1['depth_thr'] and max_depth_interval > key1['depth_interval_thr'] and spon_frame_dur> key1['spon_framedur_thr']:
            
            key1['ndepth'] = len(udepth)
            key1['field_depth'] = udepth
            key1['spon_frame_dur'] = spon_frame_dur
            self.insert1(key1)

@schema
class SpontaneousActivity(dj.Computed):
    definition = """
    -> BorderRestrict
    -> shared.SpikeMethod
    -> anatomy.Area
    -> anatomy.Layer
    ---  
    unit_number                      : int           # number of units
    unit_ids                         : blob           # list of unit ids
    mean_activity                    :  external-deeplab   # timesamples 
    activity_matrix                  :  external-deeplab   # timesamples x number of units
    spa_ctime = CURRENT_TIMESTAMP    : timestamp     # automatic    
    """
    @property
    def key_source(self):
        return BorderRestrict&LayerMembership.proj()&SponScanSel.proj()
    
    
    def make(self, key):
        units = (AreaMembership.Unit*LayerMembership.Unit)&BorderRestrict.Unit&key
        a = (dj.U('brain_area','layer')&units).fetch()
        brain_areas, layers = zip(*a)
        n = len(a)
        
        for i in range(n):
            ba = brain_areas[i]
            layer = layers[i]            
            outkey = key.copy()
            
            area_keys = units&{'brain_area':ba,'layer':layer}
            outkey['brain_area'] = ba
            outkey['layer'] = layer
            outkey['unit_number'] = len(area_keys)
            outkey['spike_method']=5
            outkey['unit_ids'] = area_keys.fetch('unit_id')
            Trace = (meso.Activity.Trace&area_keys & {'spike_method':5}).fetch('trace')
            spon_start_idx = (Sponscan&key).fetch('spon_frame_start')[0]
            m = np.vstack(Trace).T
            m = m[spon_start_idx:,:] 
            outkey['activity_matrix'] = m        
            mactivity = np.mean(m,1)
            outkey['mean_activity'] = mactivity    
            dj.conn()
            self.insert1(outkey)

#popout = SpontaneousActivity.populate(order="random",display_progress=True,suppress_errors=True)












