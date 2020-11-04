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
from sang_neuro import AreaMembership, BorderRestrict, LayerMembership, BorderDistance
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
            outkey['unit_ids'] = area_keys.fetch('unit_id', order_by='unit_id ASC')
            Trace = (meso.Activity.Trace&area_keys & {'spike_method':5}).fetch('trace', order_by='unit_id ASC')
            spon_start_idx = (Sponscan&key).fetch('spon_frame_start')[0]
            m = np.vstack(Trace).T
            m = m[spon_start_idx:,:] 
            outkey['activity_matrix'] = m        
            mactivity = np.mean(m,1)
            outkey['mean_activity'] = mactivity    
            dj.conn()
            self.insert1(outkey)

#popout = SpontaneousActivity.populate(order="random",display_progress=True,suppress_errors=True)

from util.sigproc import NaNSpline, load_eye_traces, load_frame_times_vstim, load_frame_times_behav, load_treadmill_trace
from util.sigproc import get_filter

@schema
class DownSampleDur(dj.Lookup):
    definition = """
    # window size to smooth behavior trace
    duration      :float   # downsampling duration
    ---
    
    """
    contents = [{'duration':0},
                {'duration':0.125},
                {'duration':0.25},
                {'duration':0.5},
                {'duration':1},
                {'duration':2}
                ]

@schema
class Behav4Spon(dj.Computed):
    definition="""
    # behav trace during spontaneous recording
    
    ->pupil.Eye
    ->treadmill.Treadmill
    ->DownSampleDur.proj(behav_down_dur = 'duration')    # duration for behavior down-sampling
    ---
    quantile_list         : external-deeplab        # quantile list during entire scan
    pupil_stat            : external-deeplab        # activity level from quantile list
    treadmill_stat        : external-deeplab        # activity level from quantile list
    t                     : external-deeplab        # sampled timestamps in behavior clock
    pupil_radius          : external-deeplab        # pupil radius
    treadmil_absvel       : external-deeplab        # treadmil absolute velocity
    """
    @property
    def key_source(self):
        return (dj.U('animal_id','session','scan_idx')&SpontaneousActivity)*DownSampleDur.proj(
                behav_down_dur='duration')&'behav_down_dur>0'
    
    def make(self, scankey):
        v, treadmill_time = load_treadmill_trace(scankey)
        #ftimes_vstim = load_frame_times_vstim(scankey) #2p-frame time on visual stimulus clock
        ftimes_behav = load_frame_times_behav(scankey)  #2p-frame time on behavior clock
        radius, xy, eye_time, tracking_method = load_eye_traces(scankey, shape='circle')
        
        
        outkey = scankey.copy()
        dur = outkey['behav_down_dur']  # filtering duration
        
        
        # filtering for treadmill velocity
        treadmill_sp = np.nanmedian(np.diff(treadmill_time)) # sampling period
        filter_shape = get_filter(dur, treadmill_sp)
        smooth_v = np.convolve(v,filter_shape, mode='same')
        
        
        # filtering for eye
        eye_sp = np.nanmedian(np.diff(eye_time)) # sampling period
        filter_shape = get_filter(dur, eye_sp)
        smooth_pupil = np.convolve(radius,filter_shape, mode='same')
        
        # NaNSpline allows to sample velocity at arbitary timepoints 
        # between min(treadmill_time) and max(treadmill_time) 
        treadmill_spline = NaNSpline(treadmill_time, smooth_v,k=1, ext=0) 
        pupil_spline = NaNSpline(eye_time, smooth_pupil, k=1, ext=0)
        
        
        spon_start_idx = (Sponscan&scankey).fetch('spon_frame_start')[0]
        spon_start_time = ftimes_behav[spon_start_idx]
        spon_end_time = ftimes_behav[-1]
        
        # behavior trace for the entire scantime
        t1 = np.arange(ftimes_behav[0], spon_end_time,dur/2)
        pup1 = pupil_spline(t1)
        tread_v1 = treadmill_spline(t1)
        
        a= np.vstack([pup1, tread_v1])
        quantile_list = np.arange(0,1.05,0.1)
        stat = np.nanquantile(np.abs(a),quantile_list,axis=1)
        outkey['quantile_list'] = quantile_list
        outkey['pupil_stat'] = stat[:,0]
        outkey['treadmill_stat'] = stat[:,0]

        t = ftimes_behav[spon_start_idx:]
        pup = pupil_spline(t)
        tread_v = treadmill_spline(t)
        outkey['t'] = t
        outkey['pupil_radius'] = pup
        outkey['treadmil_absvel'] = abs(tread_v)
        dj.conn()
        self.insert1(outkey)

        
#popout = Behav4Spon.populate(order="random",display_progress=True,suppress_errors=True)        
#popout = Behav4Spon.populate(display_progress=True)      
@schema
class BehavMarker(dj.Lookup):
    definition="""
    # Behavior Marker
    marker       :       char(20)   # behavior marker
    ---
    """
    contents=[['pupilR'],       # pupil radius
              ['Grad_pupilR'],  # Gradient of pupil radius
              ['PGrad_pupilR'], # Positive Gradient of pupil radius
              ['NGrad_pupilR'], # Negative Gradient of pupil radius
              ['TreadV']]       # Velocity of Treadmill

from functools import partial
@schema
class CorrBehav2Spon(dj.Computed):
    definition ="""
    ->BorderRestrict
    ->Behav4Spon
    ->DownSampleDur.proj(window_size = 'duration')
    --- 
    beh_markers      : external-deeplab     # behavior markers
    brain_areas      : external-deeplab     # brain_area list
    layers           : external-deeplab      # layer list    
    nunit            : external-deeplab      # list of nunit
    corr_mat         : external-deeplab      # correlation coefficent beh_markers x rois   
    """
    @property
    def key_source(self):
        return  (BorderRestrict*Behav4Spon)*DownSampleDur.proj(window_size='duration')&'window_size>=0.5 or window_size=0'

    def make(self, key):
        outkey = key.copy()
        print(key)
        
        ba, layer, nunit, Pact = (SpontaneousActivity&key).fetch('brain_area','layer', 'unit_number','mean_activity')
        Pact = np.vstack(Pact)
        t, pR, tV =  (Behav4Spon&key).fetch('t','pupil_radius','treadmil_absvel')
        pRd = np.gradient(pR[0],t[0])
        pRdp =pRd.copy()
        pRdn =pRd.copy()
        pRdp[pRdp<0]=0
        pRdn[pRdn>0]=0
        pRdn = np.abs(pRdn)
        
        min_ = min(len(t[0]), Pact.shape[1])
        X =np.vstack((Pact[:,:min_],pR[0][:min_],pRd[:min_], pRdp[:min_],
                      pRdn[:min_], tV[0][:min_]))
        list_beh_marker = ['pupilR','Grad_pupilR','PGrad_pupilR','NGrad_pupilR','TreadV']
        outkey['beh_markers']= ", ".join(list_beh_marker)
        
        
        windowsize = key['window_size']
        if windowsize > 0:
            sampling_period = np.nanmedian(np.diff(t[0]))
            h = get_filter(windowsize,sampling_period,'avg')
            fun = partial(np.convolve,v=h, mode='same')
            X_ = list(map(fun,X))
            X = np.vstack(X_)
            X = X[:,::len(h)]
        
        idx  = ~np.isnan(X)
        idx = np.all(idx, axis=0)
        corr = np.corrcoef(X[:,idx])
        nroi = len(ba)
        corr_ = corr[nroi:,:nroi]
        
        
        outkey['brain_areas'] = ", ".join(ba)
        outkey['layers'] = ", ".join(layer)
        outkey['nunit'] = nunit
        outkey['corr_mat']=corr_
        dj.conn()
        
        self.insert1(outkey)
        
    @staticmethod
    def collect_corr(behav_down_dur, window_size,  marker_list=[],border_distance_um=50):
        """
        plot correlation coef between population activity and beh_marker
        e.g. 
        behav_down_dur=0.5
        window_size = 2
        marker_list=['pupilR','TreadV']
        """
        
#        marker_list=['Grad_pupilR','PGrad_pupilR','NGrad_pupilR']
        
        
        criterion = {'behav_down_dur':behav_down_dur,\
                     'window_size':window_size,\
                     'border_distance_um': border_distance_um,\
                     'edge_distance_px': 10}
        if len(marker_list)==0:
            marker_list = BehavMarker.fetch('marker')
        
        
        aid, sess, sidx, ma, ba, l, corrmat = (CorrBehav2Spon&criterion).fetch(
                'animal_id','session','scan_idx','beh_markers','brain_areas','layers','corr_mat')   
        scan_info = pd.DataFrame({'aid':aid,'ses':sess,'scan':sidx})
        scan_list = scan_info.groupby(['aid','ses']).indices
        
     
        
        col = []
        for x,y in zip(ba,l):
            x=(x[0].split(', '))
            y=(y[0].split(', '))
            col.extend(list(zip(x,y)))
        col = list(set(col))        
        nrw = len(marker_list)        
        R = np.full((nrw,len(col), len(scan_list)), fill_value = np.nan) # average across scans with a session
        R1 = np.full((nrw,len(col), len(corrmat)), fill_value = np.nan) # not average across scans
        
        search1 = lambda A,x: [k for k,a in enumerate(A) if a==x ]
        
        def search_col(x):
            out = search1(col, x)
            assert len(out)==1, 'col should be a set'
            return out[0]
        
        
        for i, p in enumerate((scan_list.keys())):
            scanidx = scan_list[p]
            corr_ = corrmat[scanidx]      
            #print(scanidx)
            
            R_ = np.full((nrw,len(col), len(scanidx)), fill_value = np.nan)
            for j, scani in enumerate(scanidx):
                corr_ = corrmat[scani]
                bal = list(zip(ba[scani][0].split(', '),l[scani][0].split(', ')))
                col_idx = list(map(search_col,bal))
                #print(col_idx) #print(np.array(col)[col_idx])
                ma_ = ma[scani][0].split(', ')
                
                def search_marker_list(x):
                    out = search1(ma_, x)
                    assert len(out)==1, 'ma_ should be a set'
                    return out[0]
                ma_idx = list(map(search_marker_list, marker_list)) 
                R_[:,col_idx,j] = corr_[ma_idx]
            R[:,:,i] = np.nanmean(R_, axis=2)
            R1[:,:,scanidx] = R_
            
            
        return R, R1, col, marker_list

    @staticmethod                        
    def plot_corr(R, col, row, thr_n=5):    
        xtick_str =np.array([a[:2]+'\n'+b for a, b in col])
        mR = np.nanmean(R, axis=2)
        nR = np.sum(abs(R)>0,axis=2)
        eR = np.nanstd(R, axis=2)/np.sqrt(nR)
        mR[nR<thr_n]=0
        eR[nR<thr_n]=0
        
        
        
        xaxis = np.arange(len(xtick_str))
        hfig = plt.figure(figsize=(10,5*len(row)))
        for i in range(len(marker_list)):            
            plt.subplot(len(row),1,i+1)
            cid = np.argsort(mR[i])[::-1]                    
            plt.errorbar(xaxis, mR[i,cid],eR[i,cid])
            plt.title(row[i])
            plt.xticks(xaxis,xtick_str[cid])
            plt.plot(xaxis, np.zeros(xaxis.shape),color='k', linestyle ='--', linewidth=1)
            
#popout = CorrBehav2Spon.populate(display_progress=True)
#popout = CorrBehav2Spon.populate(order="random",display_progress=True,suppress_errors=True)
     
behav_down_dur=0.5
window_size = 0.5
marker_list=['pupilR','Grad_pupilR','TreadV']            
R, R1, col, marker_list = CorrBehav2Spon.collect_corr(behav_down_dur, window_size,  marker_list) 

           
                
CorrBehav2Spon.plot_corr(R, col, marker_list)                
  
            
        
        



#



