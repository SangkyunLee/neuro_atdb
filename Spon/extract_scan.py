#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:58:03 2020

@author: admin
"""
# extracting mesoscan data with maximum depth info and stimulus repetition


import pandas as pd



key_info  =(dj.U('animal_id','session','scan_idx')& meso.ScanDone).fetch()
import time

Out = pd.DataFrame()
i =0
for aid,sid,scan_id in key_info:

    
    skey = {'animal_id':aid, 'session':sid,'scan_idx':scan_id}
    print(skey)
    s = meso.ScanDone &skey
    s1 = meso.StackCoordinates.UnitInfo &skey
    depthinfo  = (s.aggr(s1, max_z ="max(stack_z)"))
    s2= dj.U('animal_id','session','scan_idx').aggr(depthinfo,maxz='max(max_z)')
    scan_depth = s2.fetch(as_dict=True)
    print(scan_depth)
    # time.sleep(0.02)    
    
    if len(scan_depth)>0:
        
            
        cond  = dj.U('stimulus_type','condition_hash') & ((stimulus.Trial&skey)*stimulus.Condition)    
        nrep  = pd.DataFrame(cond.aggr((stimulus.Trial&skey), repeat= 'count(*)').fetch())
        nrep = nrep.iloc[:,1:]
        #nrep = nrep.sort_values(['repeat'],ascending=False)
        nrep1 = nrep.groupby('stimulus_type').max()
        nrep1 = nrep1.rename(columns={'repeat':'max_repeat'})
        
        
        a=nrep1.to_dict()
        b=dict(scan_depth[0])
        out_= {**b,**a}
        Out = Out.append(out_,ignore_index=True)    
        print('###############i:{}_{}:{}'.format(i,aid,sid))
        i+=1
    


cols=['animal_id','session','scan_idx','maxz','max_repeat']
Out = Out[cols]
Out.to_csv('./doc/meso_stim_perscan.csv') 


# select c.stimulus_type from trial t, condition c
# where t.animal_id=18142 and t.session=3 ;     

# A=dj.U('animal_id','session').aggr(meso.ScanDone*meso.StackCoordinates.UnitInfo,maxz='max(stack_z)')
# A & 'maxz>500'
select s.animal_id, s.session, s.scan_idx, max(info.stack_z) from __scan_done s,  __stack_coordinates__unit_info  info
where s.animal_id=info.animal_id and s.session=info.session
group by info.animal_id,info.session,info.scan_idx having max(info.stack_z)>500;

##############################################
# loading spontaneous data info
##################################333

A=dj.U('animal_id','session','scan_idx').aggr(meso.ScanDone*meso.StackCoordinates.UnitInfo,maxz='max(stack_z)')
keyinfo = (dj.U('animal_id','session','scan_idx')&(A & 'maxz>450')).fetch()


Spon = pd.DataFrame()

for aid,ses,iscan in keyinfo:
    
    key = {'animal_id': aid, 'session':ses, 'scan_idx':iscan}
        
    flip_times = (stimulus.Trial & key).fetch('flip_times')
    last_flip_times = flip_times[-1].squeeze()[-1]
    
    if len(stimulus.Sync & key)>0:
        frame_times = (stimulus.Sync & key).fetch1('frame_times').squeeze() 
        slice_num = len(np.unique((meso.ScanInfo.Field & key).fetch('z')))
        field_offset=0
        frame_times = frame_times[field_offset::slice_num]
        
        spon_frame_start = np.where(frame_times>last_flip_times)[0][0]
        spon_dur = len(frame_times)-spon_frame_start
        b= {'spon_frame_start':spon_frame_start, 'spon_framedur':spon_dur}
        out_= {**key, **b}
        print(out_) 
        Spon= Spon.append(out_,ignore_index=True)
cols=['animal_id','session','scan_idx','spon_frame_start','spon_framedur']
Spon = Spon[cols]

stiminfo = pd.read_csv('./doc/meso_stim_perscan.csv')
out = stiminfo.merge(Spon)
out.to_csv('./doc/meso_stim_spon_perscan.csv') 







# case 1 
###### stimulus.Trial &'animal_id=17797,session=6, t.scan_idx=6' * stimulus.Condtion
# select t.animal_id, t.session, t.scan_idx, t.trial_idx, t.condition_hash, c.stimulus_type 
# from trial  t 
# left join pipeline_stimulus.condition c 
# on t.condition_hash = c.condition_hash
# where t.animal_id=17797 and t.session=6 and t.scan_idx=6 ;



# case 2:
# stimulus.Trial * stimulus.Condtion
# will result valid number on 'animal_id=17797,session=6, t.scan_idx=6' 
#########################################3
# select t.animal_id, t.session, t.scan_idx, t.trial_idx, t.condition_hash, c.stimulus_type 
# from trial  t 
# left join pipeline_stimulus.condition c 
# on t.condition_hash = c.condition_hash
# and t.animal_id=17797 and t.session=6 and t.scan_idx=6 ;
