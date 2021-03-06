#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:36:38 2020

@author: sang
"""

#
#%run access_db.py
#runfile('access_db.py')

import datajoint as dj
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from pipeline import experiment, reso, meso, fuse, stack,  treadmill, pupil, shared
#from stimulus import stimulus
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')  

#experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
#meso = dj.create_virtual_module('meso', 'pipeline_meso')
#pupil = dj.create_virtual_module('pupil', 'pipeline_eye')
#stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
#treadmill = dj.create_virtual_module('treadmill', 'pipeline_treadmill')
# shared = dj.create_virtual_module('shared', 'pipeline_shared')



# schema = dj.schema('sang_neuro', locals(), create_tables=True)
schema = dj.schema('sang_neuro')


            
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
        return (meso.ScanInfo*shared.SegmentationMethod) & anatomy.AreaMask & meso.ScanDone& {'pipe_version':1, 'segmentation_method':6} 
        #return key1 #& {'animal_id': 17797, 'session': 6, 'scan_idx': 4}
    
    class Unit(dj.Part):
        definition = """ 
        -> master      
        unit_id                               : int          # unit id
        ---
        -> anatomy.Area        
        """

    def make(self, key):
        fields = (meso.ScanInfo.Field & anatomy.AreaMask &(meso.ScanDone& key)).fetch('field')
        
        #print(key)
        #print(len(fields))
        if len(fields)>0:                   
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
            
            
#popout = AreaMembership.populate(reserve_jobs=True,display_progress=True,
#                                 suppress_errors=True,
#                                 return_exception_objects=True,
#                                 order="random")           

          
def exclude_border_neighbor(unit_loc_pix, border_loc_pix, field_shape, microns_per_pixel=(1,1),
                           px_dist_edge=10, dist_border=30):
    """
    remove units close to field edge and area borders
    
    arguments:
    unit_loc_pix : tuple of lists of y and x in pixel location of units
    border_loc_pix : tuple of lists of y and x in pixel location of area border
                    if no border is given within a field,  ([],[])
    field_shape : tupe of y- and x- size
    px_dist_edge : pixel distance to field edge
    dist_border : distance(um) threshold to area border 
    
    return:
        cidx[0]   : select unit ids
        cmask     : mask to display units with borders
        
    """
    import numpy as np
    px_y, px_x = unit_loc_pix
    
    # edge mask
    mask = np.ones(field_shape)*True
    mask[slice(px_dist_edge),:]=False
    mask[slice(field_shape[0]-px_dist_edge, field_shape[0]),:]=False
    mask[:,slice(px_dist_edge)]=False
    mask[:,slice(field_shape[1]-px_dist_edge, field_shape[1])]=False
    #return mask
    if len(border_loc_pix[0])>0:

        df_y=  px_y[:,np.newaxis] -  border_loc_pix[0]
        df_x=  px_x[:,np.newaxis] -  border_loc_pix[1]    
        
        dist = np.sqrt((df_y*microns_per_pixel[0])**2+(df_x*microns_per_pixel[1])**2)
        # minium distance to any area border
        mindist = np.min(dist,axis=1)
        cidx  = np.where(np.logical_and(mindist>dist_border, mask[px_y, px_x]))
    else: # when bodrder_loc_pix is empty
        cidx  = np.where(mask[px_y, px_x])
        
    cmask = np.zeros(field_shape)    
    cmask[border_loc_pix[0],border_loc_pix[1]] = 1    
    cmask[px_y[cidx[0]], px_x[cidx[0]] ]=2
    
    return cidx[0], cmask


def remove_overlap_field(field_center, field_size, field_ids):
    """ remove overlap area between fields, always overlap removed in the later field number
    
    field_center: x, y, z, tuple of field_list
    field_size: height, width, tuple of field_list
    field_ids: list of field id
    
    return:
        select_range:    list of field_range [[y0,x0], [y1, x1],field_order] 
        field_ids:       list of field ids, i.e., field_ids[field_order]
    """
    import numpy as np
    
    x, y, z = field_center
    height, width = field_size 
    
    select_range = []
    fz =[]
    for fid in range(len(x)):
        y0 = round(y[fid] - height[fid]/2)
        y1 = round(y[fid] + height[fid]/2)
        x0 = round(x[fid] - width[fid]/2)
        x1 = round(x[fid] + width[fid]/2)
        
        lt = np.array([y0,x0]) # left-top corner
        rb = np.array([y1,x1]) #right-bottom corner
        if z[fid] in fz:
            ix = np.where(fz==z[fid])[0][0]
            for j in range(len(select_range[ix])):
                rb_ = select_range[ix][j][1]
                iou = rb_ -lt
                if np.all(iou>0): # if two fields are overlap
                    
                    # find directions of iou are smaller than field size
                    overlap_direction = np.where (([height[fid], width[fid]] -iou)>0)[0] 
                    lt[overlap_direction] = rb_[overlap_direction]
                    
            select_range[ix].append([lt, rb, fid])
        else:
            fz.append(z[fid])
            select_range.append([[lt, rb, fid]])
            
    out = []    
    for i in range(len(select_range)):
        out += select_range[i]
        
    flist=[i[2] for i in out]
    
    return out, field_ids[flist]

@schema
class BorderDistance(dj.Lookup):
    definition="""
    # distance of unit to anatomy area borders
    bd_distance       :       float   # distance to Border
    ---
    """
    contents = [{'bd_distance':30},
                {'bd_distance':50},
                {'bd_distance':70},
                {'bd_distance':100}
                ]

@schema
class BorderRestrict(dj.Computed):
    definition=""" # remove units close to AreaBorder and field edge
    -> AreaMembership    
    edge_distance_px                      : smallint    # pixel distance to field edge
    border_distance_um                    : float  # distance (um) to Areaborder
    field_overlap_remove                        : tinyint   # boolean to decide whether overalp field is removed
    --- 
    caclulation_time=CURRENT_TIMESTAMP    : timestamp     # automatic  
    """
    
    class Unit(dj.Part):
        definition = """ 
        -> master      
        unit_id                               : int          # unit id
        ---    
        um_x                : smallint      # x-coordinate of centroid in motor coordinate system
        um_y                : smallint      # y-coordinate of centroid in motor coordinate system
        um_z                : smallint      # z-coordinate of mask relative to surface of the cortex
        """
    class FieldMask(dj.Part):
        definition = """
        ->master
        field                              : int          # field id
        ---
        unit_image                         : blob          # area border and unit
        """
        
    def make(self, key):
        
        key1 = key.copy()
        key1['edge_distance_px'] = 10        
        key1['field_overlap_remove'] = 1
        
        
        field_ids, um_height, um_width,fx0, fy0, fz0 = (meso.ScanInfo.Field&key).fetch(
                'field', 'um_height', 'um_width', 'x', 'y', 'z')        
        
        if key1['field_overlap_remove']>0:
            field_center = (fx0, fy0, fz0)
            field_size = (um_height, um_width)
            field_restriction, field_order = remove_overlap_field(field_center, field_size, field_ids)
        
        for bdist in BorderDistance.fetch('bd_distance'):
        
            key1['border_distance_um'] = bdist
            self.insert1(key1)
            print(key1)
            for ifid, fid in enumerate(field_ids):                
                field_key = key1.copy()
                field_key['field']=fid
                
                 #units from a specific field
                unit_ids, px_x, px_y, um_x, um_y = (meso.ScanSet.UnitInfo &
                                        (meso.ScanSet.Unit& field_key)).fetch('unit_id','px_x','px_y', 'um_x','um_y')
                
                
                # create area mask
                area_masks, areas =(anatomy.AreaMask & field_key).fetch('mask','brain_area')
                area_mask = np.zeros(area_masks[0].shape)
                
                for iarea in range(len(area_masks)):                    
                    area_mask[area_masks[iarea]>0] = iarea+1
                    
                import cv2
                edge_image = cv2.Laplacian(area_mask,cv2.CV_64F)
                border_loc_pix  = np.where(abs(edge_image)>0) 
                
                if len(border_loc_pix)<2 or len(border_loc_pix[0])==0: # when single area is imaged
                    border_loc_pix = ([],[])                       
                
                microns_per_pixel = (meso.ScanInfo.Field& field_key).microns_per_pixel            
                uix, mask_img = exclude_border_neighbor((px_y,px_x), border_loc_pix, 
                                                       area_mask.shape, 
                                                       microns_per_pixel,
                                                       field_key['edge_distance_px'], 
                                                       field_key['border_distance_um'])
               
                
                unit_ids_sel= unit_ids[uix]
                
                if key1['field_overlap_remove']>0:                    
                    xs = um_x[uix]
                    ys = um_y[uix]
                    inx = int(np.where(field_order==fid)[0])
                    frestrict = field_restriction[inx]
                    conds = np.vstack((frestrict[0][0]<ys, ys<frestrict[1][0], frestrict[0][1]<xs, xs<frestrict[1][1]))
                    uix2 = np.where( conds.all(axis=0))[0]                
                    unit_ids_sel = unit_ids_sel[uix2]
                
                
                
                field_key['unit_image']= mask_img.astype('int8')
                dj.conn()
                BorderRestrict.FieldMask.insert1(field_key)
                
                dj.conn()
                for iu in unit_ids_sel:
                    unit_key = key1.copy()
                    unit_key['unit_id']= iu
                    ix  = np.where(unit_ids==iu)[0]
                    unit_key['um_z'] = int(fz0[ifid])
                    unit_key['um_x'] = um_x[ix][0]
                    unit_key['um_y'] = um_y[ix][0]
                    BorderRestrict.Unit.insert1(unit_key)
                print(unit_key)
                
#popout = BorderRestrict.populate(order="random",display_progress=True,suppress_errors=True)
#

@schema            
class LayerMembership(dj.Computed):
    definition = """ # this is a replicate of anatomy.AreaMembership to populate mylist
    -> meso.ScanInfo
    -> shared.SegmentationMethod
    ---    
    caclulation_time=CURRENT_TIMESTAMP    : timestamp     # automatic    
    """
    
    class Unit(dj.Part):
        definition = """ 
        -> master      
        unit_id                               : int          # unit id
        ---
        -> anatomy.Layer        
        """
    @property
    def key_source(self):
        return (meso.ScanInfo*shared.SegmentationMethod) & meso.ScanDone& {'pipe_version':1, 'segmentation_method':6} 

    def make(self, key):
        field_keys = ((meso.ScanInfo.Field*shared.SegmentationMethod) &(meso.ScanDone& key)).fetch('KEY')
        self.insert1(key)
        
        layers, zstart, zend = anatomy.Layer.fetch('layer','z_start','z_end')
            
        for field_key in field_keys:
            depth = (meso.ScanInfo.Field &field_key).fetch('z')
             
            mask_keys = (meso.ScanSet.Unit&field_key).fetch('KEY')
            
            idx = np.all(np.vstack((zstart<depth[0], zend>=depth[0])),0)
            if len(np.where(idx))==1:
                layer = layers[idx][0]
            else:
                layer = 'unset'
                
            for i in range(len(mask_keys)):
                mask_keys[i]['layer']=layer
            dj.conn()
            
            LayerMembership.Unit.insert(mask_keys)

#popout = LayerMembership.populate(display_progress=True)
