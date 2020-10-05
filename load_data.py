#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:33:23 2020

@author: sang
"""
# import importlib
# importlib.reload(dj)

import datajoint as dj
import os
 
dj.config['database.host'] = os.environ.get('ATHOST')
dj.config['database.user'] = os.environ.get('ATUSER')
dj.config['database.password'] = os.environ.get('ATPW')

         
dj.conn()
 
from pipeline import experiment, reso, meso, fuse, stack,  treadmill, pupil, shared
from stimulus import stimulus
from stimulus.utils import get_stimulus_info

anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')    