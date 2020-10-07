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

#experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
#meso = dj.create_virtual_module('meso', 'pipeline_meso')
#pupil = dj.create_virtual_module('pupil', 'pipeline_eye')
#stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
#treadmill = dj.create_virtual_module('treadmill', 'pipeline_treadmill')
# shared = dj.create_virtual_module('shared', 'pipeline_shared')



# schema = dj.schema('sang_neuro', locals(), create_tables=True)
schema = dj.schema('sang_neuro')

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
            
# *animal_id    *session    *scan_idx    *pipe_version  *segmentation_ *unit_id    brain_area             
@schema            
class AreaMembership(dj.Computed):
    definition = """ # this is a replicate of anatomy.AreaMembership to populate mylist
    -> meso.ScanInfo
    -> shared.SegmentationMethod
    ---    
    caclulation_time=CURRENT_TIMESTAMP    : timestamp     # automatic    
    """
    @property
    def key_source(self):
        return (meso.ScanInfo*shared.SegmentationMethod) & anatomy.AreaMask & {'pipe_version':1, 'segmentation_method':6} 
        #return key1 #& {'animal_id': 17797, 'session': 6, 'scan_idx': 4}
    
    class Unit(dj.Part):
        definition = """ 
        -> master      
        unit_id                               : int          # unit id
        ---
        -> anatomy.Area        
        """
#    key  = {'animal_id': 17797, 'session': 6, 'scan_idx': 4, 'pipe_version': 1, 'segmentation_method': 6}
    def make(self, key):
        
        print(key)
        
        fields = (meso.ScanInfo.Field & anatomy.AreaMask &(meso.ScanDone& key)).fetch('field')
        #field_keys = (meso.ScanSet & key).fetch('KEY')
        
        
        self.insert1(key)
        for field_id in fields:
            field_key = key.copy()
            field_key['field'] = field_id
            area_masks, areas =(anatomy.AreaMask & field_key).fetch('mask','brain_area')
            
            area_mask = np.nan*np.ones(area_masks[0].shape)
            for iarea in range(len(area_masks)):
                area_mask[area_masks[iarea]>0] = iarea
            
            
            #units selected from a specific field
            unit_ids, px_x, px_y = (meso.ScanSet.UnitInfo & 
                                    (meso.ScanSet.Unit& field_key)).fetch('unit_id','px_x','px_y')
                       
            dj.conn()
            for i, uid in enumerate(unit_ids):                
                area_idx = (area_mask[round(px_y[i]),round(px_x[i])])
                tup_out = key.copy()            
                tup_out['unit_id'] = uid                
                if np.isnan(area_idx):
                    tup_out['brain_area']= 'unknown'                    
                else:                    
                    tup_out['brain_area']= areas[int(area_idx)]                
                AreaMembership.Unit.insert1(tup_out)
            
            
popout = AreaMembership.populate(reserve_jobs=True,display_progress=True,
                                 suppress_errors=True,
                                 return_exception_objects=True,
                                 order="random")           
            

#key = {'animal_id': 17358, 'session': 1, 'scan_idx': 13,'segmentation_method':6,'spike_method':5}
                


#aunit_id, brain_area = (AreaMembership.Unit & key).fetch('unit_id','brain_area')
#funit_id, trace = (meso.Activity.Trace&key).fetch('unit_id','trace')

## SponScanSel.populate()            
#key = (meso.ScanDone&SponScanSel&anatomy.AreaMembership).fetch('KEY')[0] 
#
#
#key1 =  {'animal_id': 17358, 'session': 1, 'scan_idx': 13, 'pipe_version': 1, 'field': 1}
#key1 =  {'animal_id': 17358, 'session': 1, 'scan_idx': 13, 'pipe_version': 1, 'segmentation_method': 6}
#
#experiment.Scan * experiment.Session & key1
#field_key = ((fuse.ScanSet& anatomy.AreaMask) & key1).fetch('KEY')[0]
#area_masks, areas = (anatomy.AreaMask & field_key).fetch('mask','brain_area')
#
#
#
#meso.ScanSet.UnitInfo &field_key