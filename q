[1mdiff --git a/Spon/.gitignore b/Spon/.gitignore[m
[1mdeleted file mode 100755[m
[1mindex ee4c926..0000000[m
[1m--- a/Spon/.gitignore[m
[1m+++ /dev/null[m
[36m@@ -1 +0,0 @@[m
[31m-/test[m
[1mdiff --git a/Spon/.ipynb_checkpoints/Untitled-checkpoint.ipynb b/Spon/.ipynb_checkpoints/Untitled-checkpoint.ipynb[m
[1mdeleted file mode 100644[m
[1mindex 7fec515..0000000[m
[1m--- a/Spon/.ipynb_checkpoints/Untitled-checkpoint.ipynb[m
[1m+++ /dev/null[m
[36m@@ -1,6 +0,0 @@[m
[31m-{[m
[31m- "cells": [],[m
[31m- "metadata": {},[m
[31m- "nbformat": 4,[m
[31m- "nbformat_minor": 4[m
[31m-}[m
[1mdiff --git a/Spon/Untitled.ipynb b/Spon/Untitled.ipynb[m
[1mdeleted file mode 100644[m
[1mindex 7fa5dec..0000000[m
[1m--- a/Spon/Untitled.ipynb[m
[1m+++ /dev/null[m
[36m@@ -1,119 +0,0 @@[m
[31m-{[m
[31m- "cells": [[m
[31m-  {[m
[31m-   "cell_type": "code",[m
[31m-   "execution_count": 1,[m
[31m-   "metadata": {},[m
[31m-   "outputs": [[m
[31m-    {[m
[31m-     "name": "stdout",[m
[31m-     "output_type": "stream",[m
[31m-     "text": [[m
[31m-      "Connecting sang@at-database.ad.bcm.edu:3306\n",[m
[31m-      "Loading local settings from pipeline_config.json\n"[m
[31m-     ][m
[31m-    },[m
[31m-    {[m
[31m-     "name": "stderr",[m
[31m-     "output_type": "stream",[m
[31m-     "text": [[m
[31m-      "/home/admin/anaconda3/envs/dj/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",[m
[31m-      "  _np_qint8 = np.dtype([(\"qint8\", np.int8, 1)])\n",[m
[31m-      "/home/admin/anaconda3/envs/dj/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",[m
[31m-      "  _np_quint8 = np.dtype([(\"quint8\", np.uint8, 1)])\n",[m
[31m-      "/home/admin/anaconda3/envs/dj/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",[m
[31m-      "  _np_qint16 = np.dtype([(\"qint16\", np.int16, 1)])\n",[m
[31m-      "/home/admin/anaconda3/envs/dj/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",[m
[31m-      "  _np_quint16 = np.dtype([(\"quint16\", np.uint16, 1)])\n",[m
[31m-      "/home/admin/anaconda3/envs/dj/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",[m
[31m-      "  _np_qint32 = np.dtype([(\"qint32\", np.int32, 1)])\n",[m
[31m-      "/home/admin/anaconda3/envs/dj/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",[m
[31m-      "  np_resource = np.dtype([(\"resource\", np.ubyte, 1)])\n"[m
[31m-     ][m
[31m-    },[m
[31m-    {[m
[31m-     "name": "stdout",[m
[31m-     "output_type": "stream",[m
[31m-     "text": [[m
[31m-      "DLC loaded in light mode; you cannot use the labeling GUI!\n",[m
[31m-      "DLC loaded in light mode; you cannot use the relabeling GUI!\n"[m
[31m-     ][m
[31m-    }[m
[31m-   ],[m
[31m-   "source": [[m
[31m-    "%run access_db.py"[m
[31m-   ][m
[31m-  },[m
[31m-  {[m
[31m-   "cell_type": "code",[m
[31m-   "execution_count": 2,[m
[31m-   "metadata": {},[m
[31m-   "outputs": [],[m
[31m-   "source": [[m
[31m-    "import datajoint as dj\n",[m
[31m-    "import numpy as np\n",[m
[31m-    "import pandas as pd\n",[m
[31m-    "import matplotlib.pyplot as plt\n",[m
[31m-    "from pipeline import experiment, reso, meso, fuse, stack,  treadmill, pupil, shared\n",[m
[31m-    "from stimulus import stimulus\n",[m
[31m-    "anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')  \n",[m
[31m-    "\n"[m
[31m-   ][m
[31m-  },[m
[31m-  {[m
[31m-   "cell_type": "code",[m
[31m-   "execution_count": 3,[m
[31m-   "metadata": {},[m
[31m-   "outputs": [],[m
[31m-   "source": [[m
[31m-    "%matplotlib inline"[m
[31m-   ][m
[31m-  },[m
[31m-  {[m
[31m-   "cell_type": "code",[m
[31m-   "execution_count": 6,[m
[31m-   "metadata": {},[m
[31m-   "outputs": [[m
[31m-    {[m
[31m-     "ename": "AttributeError",[m
[31m-     "evalue": "'DiGraph' object has no attribute 'node'",[m
[31m-     "output_type": "error",[m
[31m-     "traceback": [[m
[31m-      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",[m
[31m-      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",[m
[31m-      "\u001b[0;32m<ipython-input-6-2f628dce4f5c>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;34m(\u001b[0m\u001b[0mdj\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mERD\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmeso\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mScanInfo\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m+\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdraw\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",[m
[31m-      "\u001b[0;32m~/anaconda3/envs/dj/lib/python3.6/site-packages/datajoint/erd.py\u001b[0m in \u001b[0;36mdraw\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    293\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    294\u001b[0m         \u001b[0;32mdef\u001b[0m \u001b[0mdraw\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 295\u001b[0;31m             \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimshow\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmake_image\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    296\u001b[0m             \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgca\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0maxis\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'off'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    297\u001b[0m             \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshow\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",[m
[31m-      "\u001b[0;32m~/anaconda3/envs/dj/lib/python3.6/site-packages/datajoint/erd.py\u001b[0m in \u001b[0;36mmake_image\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    287\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    288\u001b[0m         \u001b[0;32mdef\u001b[0m \u001b[0mmake_image\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 289\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimread\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmake_png\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    290\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    291\u001b[0m         \u001b[0;32mdef\u001b[0m \u001b[0m_repr_svg_\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",[m
[31m-      "\u001b[0;32m~/anaconda3/envs/dj/lib/python3.6/site-packages/datajoint/erd.py\u001b[0m in \u001b[0;36mmake_png\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    284\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    285\u001b[0m         \u001b[0;32mdef\u001b[0m \u001b[0mmake_png\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 286\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mio\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mBytesIO\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmake_dot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcreate_png\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    287\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    288\u001b[0m         \u001b[0;32mdef\u001b[0m \u001b[0mmake_image\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",[m
[31m-      "\u001b[0;32m~/anaconda3/envs/dj/lib/python3.6/site-packages/datajoint/erd.py\u001b[0m in \u001b[0;36mmake_dot\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    272\u001b[0m                 \u001b[0medge\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_color\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'#00000040'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    273\u001b[0m                 \u001b[0medge\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_style\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'solid'\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mprops\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'primary'\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0;34m'dashed'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 274\u001b[0;31m                 \u001b[0mmaster_part\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgraph\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnode\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mdest\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'node_type'\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0mPart\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mdest\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstartswith\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msrc\u001b[0m\u001b[0;34m+\u001b[0m\u001b[0;34m'.'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    275\u001b[0m                 \u001b[0medge\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_weight\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m3\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mmaster_part\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    276\u001b[0m                 \u001b[0medge\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_arrowhead\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'none'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",[m
[31m-      "\u001b[0;31mAttributeError\u001b[0m: 'DiGraph' object has no attribute 'node'"[m
[31m-     ][m
[31m-    }[m
[31m-   ],[m
[31m-   "source": [[m
[31m-    "(dj.ERD(meso.ScanInfo)+1).draw() "[m
[31m-   ][m
[31m-  }[m
[31m- ],[m
[31m- "metadata": {[m
[31m-  "kernelspec": {[m
[31m-   "display_name": "Python 3",[m
[31m-   "language": "python",[m
[31m-   "name": "python3"[m
[31m-  },[m
[31m-  "language_info": {[m
[31m-   "codemirror_mode": {[m
[31m-    "name": "ipython",[m
[31m-    "version": 3[m
[31m-   },[m
[31m-   "file_extension": ".py",[m
[31m-   "mimetype": "text/x-python",[m
[31m-   "name": "python",[m
[31m-   "nbconvert_exporter": "python",[m
[31m-   "pygments_lexer": "ipython3",[m
[31m-   "version": "3.6.12"[m
[31m-  }[m
[31m- },[m
[31m- "nbformat": 4,[m
[31m- "nbformat_minor": 4[m
[31m-}[m
[1mdiff --git a/Spon/access_db.py b/Spon/access_db.py[m
[1mdeleted file mode 100755[m
[1mindex 7877887..0000000[m
[1m--- a/Spon/access_db.py[m
[1m+++ /dev/null[m
[36m@@ -1,32 +0,0 @@[m
[31m-#!/usr/bin/env python3[m
[31m-# -*- coding: utf-8 -*-[m
[31m-"""[m
[31m-Created on Wed Sep  9 11:33:23 2020[m
[31m-[m
[31m-@author: sang[m
[31m-"""[m
[31m-# import importlib[m
[31m-# importlib.reload(dj)[m
[31m-[m
[31m-import datajoint as dj[m
[31m-import os[m
[31m- [m
[31m-dj.config['database.host'] = os.environ.get('ATHOST')[m
[31m-dj.config['database.user'] = os.environ.get('ATUSER')[m
[31m-dj.config['database.password'] = os.environ.get('ATPW')[m
[31m-[m
[31m-dj.config['external-deeplab']= dict([m
[31m-              protocol='file',[m
[31m-              location='/media/data/sang_data/.at-db_local')[m
[31m-         [m
[31m-dj.conn()[m
[31m- [m
[31m-from pipeline import experiment, reso, meso, fuse, stack,  treadmill, pupil, shared[m
[31m-from stimulus import stimulus[m
[31m-from stimulus.utils import get_stimulus_info[m
[31m-anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')  [m
[31m-        [m
[31m-[m
[31m-[m
[31m-[m
[31m-    [m
\ No newline at end of file[m
[1mdiff --git a/Spon/exp_scan_stim_stack_info.py b/Spon/exp_scan_stim_stack_info.py[m
[1mdeleted file mode 100755[m
[1mindex fc09149..0000000[m
[1m--- a/Spon/exp_scan_stim_stack_info.py[m
[1m+++ /dev/null[m
[36m@@ -1,82 +0,0 @@[m
[31m-pd.DataFrame(dj.U('animal_id') & stack.Area)[m
[31m-    [m
[31m-    animal_id[m
[31m-0       18142     good for inter-area/lamina, spon, stim repetition[m
[31m-1       20892    --> multiple brain area, only L2/3 and L4 for some scan and not repetition[m
[31m-2       20897     good for inter-area/lamina, spon, stim repetition[m
[31m-3       21067      good for inter-area/lamina, stim repetition  not for spontaneous[m
[31m-4       21154    ---> single layer image[m
[31m-5       21553    ---> single layer image[m
[31m-6       21844       ---> single layer image only L2/3[m
[31m-7       21900    ---> single layer image[m
[31m-8       22083    ---> single layer image[m
[31m-9       22279    ---> single layer image[m
[31m-10      23224       ---> single layer image only L2/3[m
[31m-11      23343    ---> single layer image[m
[31m-[m
[31m-[m
[31m-aid = 'animal_id = 20897'[m
[31m-t = (stimulus.Trial & aid )*stimulus.Condition[m
[31m-[m
[31m-out = dj.U('session','scan_idx','condition_hash','stimulus_type').aggr(t,ncount='count(stimulus_type)')[m
[31m-[m
[31m-dj.U('stimulus_type') &(out & 'ncount>9')[m
[31m-# Out[381]: [m
[31m-# *stimulus_type[m
[31m-# +------------+[m
[31m-# stimulus.Fancy[m
[31m-# stimulus.Singl[m
[31m-# stimulus.Clip [m
[31m-# stimulus.Frame[m
[31m-#  (4 tuples)[m
[31m- [m
[31m-t= dj.U('animal_id','session','scan_idx').aggr(meso.StackCoordinates.UnitInfo & aid, depth='max(stack_z)')[m
[31m- [m
[31m-#  Out[396]: [m
[31m-# *session    *scan_idx    depth         [m
[31m-# +---------+ +----------+ +------------+[m
[31m-# 3           11           234.6106414794[m
[31m-# 3           14           243.9084014892[m
[31m-# 4           11           239.9379577636[m
[31m-# 4           16           236.7787933349[m
[31m-# 5           18           240.8100891113[m
[31m-# 5           29           240.2297363281[m
[31m-# 6           17           239.7984619140[m
[31m-#    ...[m
[31m-#  (22 tuples)[m
[31m-[m
[31m-[m
[31m-keys  = (t & 'depth>350').fetch('KEY')[m
[31m-ikey =1[m
[31m-meso.ScanInfo.Field & keys[ikey]  # depth info[m
[31m-[m
[31m-#stimulus type with repetition[m
[31m-t = (stimulus.Trial & keys[ikey] )*stimulus.Condition[m
[31m-out = dj.U('session','scan_idx','condition_hash','stimulus_type').aggr(t,ncount='count(stimulus_type)')[m
[31m-[m
[31m-out &(dj.U('stimulus_type') &(out & 'ncount>9'))[m
[31m-[m
[31m-[m
[31m-[m
[31m-[m
[31m-keySource = ((experiment.Scan) *  (shared.SegmentationMethod) *  (shared.PipelineVersion) & fuse.ScanDone) &anatomy.AreaMask[m
[31m-[m
[31m-out = keySource.fetch('KEY')[m
[31m-aid = 'animal_id=18142'[m
[31m-anatomy.AreaMask&aid[m
[31m-[m
[31m-out1 = (anatomy.AreaMask&aid).fetch('KEY')[m
[31m-[m
[31m-[m
[31m-[m
[31m-aid = 'animal_id=18252'[m
[31m-keySource  & 'animal_id=18252 and pipe_version = 1'[m
[31m-[m
[31m-[m
[31m-anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')[m
[31m-[m
[31m-alist = pd.DataFrame(dj.U('animal_id') & (experiment.Scan *anatomy.Area * shared.Field) & 'animal_id = 18142'[m
[31m-[m
[31m-[m
[31m-alist.index(alist['animal_id']==18142).to_list()[m
[31m-[m
[1mdiff --git a/Spon/extract_scan.py b/Spon/extract_scan.py[m
[1mdeleted file mode 100755[m
[1mindex 35a7503..0000000[m
[1m--- a/Spon/extract_scan.py[m
[1m+++ /dev/null[m
[36m@@ -1,127 +0,0 @@[m
[31m-#!/usr/bin/env python3[m
[31m-# -*- coding: utf-8 -*-[m
[31m-"""[m
[31m-Created on Fri Sep 25 09:58:03 2020[m
[31m-[m
[31m-@author: admin[m
[31m-"""[m
[31m-# extracting mesoscan data with maximum depth info and stimulus repetition[m
[31m-[m
[31m-[m
[31m-import pandas as pd[m
[31m-[m
[31m-[m
[31m-[m
[31m-key_info  =(dj.U('animal_id','session','scan_idx')& meso.ScanDone).fetch()[m
[31m-import time[m
[31m-[m
[31m-Out = pd.DataFrame()[m
[31m-i =0[m
[31m-for aid,sid,scan_id in key_info:[m
[31m-[m
[31m-    [m
[31m-    skey = {'animal_id':aid, 'session':sid,'scan_idx':scan_id}[m
[31m-    print(skey)[m
[31m-    s = meso.ScanDone &skey[m
[31m-    s1 = meso.StackCoordinates.UnitInfo &skey[m
[31m-    depthinfo  = (s.aggr(s1, max_z ="max(stack_z)"))[m
[31m-    s2= dj.U('animal_id','session','scan_idx').aggr(depthinfo,maxz='max(max_z)')[m
[31m-    scan_depth = s2.fetch(as_dict=True)[m
[31m-    print(scan_depth)[m
[31m-    # time.sleep(0.02)    [m
[31m-    [m
[31m-    if len(scan_depth)>0:[m
[31m-        [m
[31m-            [m
[31m-        cond  = dj.U('stimulus_type','condition_hash') & ((stimulus.Trial&skey)*stimulus.Condition)    [m
[31m-        nrep  = pd.DataFrame(cond.aggr((stimulus.Trial&skey), repeat= 'count(*)').fetch())[m
[31m-        nrep = nrep.iloc[:,1:][m
[31m-        #nrep = nrep.sort_values(['repeat'],ascending=False)[m
[31m-        nrep1 = nrep.groupby('stimulus_type').max()[m
[31m-        nrep1 = nrep1.rename(columns={'repeat':'max_repeat'})[m
[31m-        [m
[31m-        [m
[31m-        a=nrep1.to_dict()[m
[31m-        b=dict(scan_depth[0])[m
[31m-        out_= {**b,**a}[m
[31m-        Out = Out.append(out_,ignore_index=True)    [m
[31m-        print('###############i:{}_{}:{}'.format(i,aid,sid))[m
[31m-        i+=1[m
[31m-    [m
[31m-[m
[31m-[m
[31m-cols=['animal_id','session','scan_idx','maxz','max_repeat'][m
[31m-Out = Out[cols][m
[31m-Out.to_csv('./doc/meso_stim_perscan.csv') [m
[31m-[m
[31m-[m
[31m-# select c.stimulus_type from trial t, condition c[m
[31m-# where t.animal_id=18142 and t.session=3 ;     [m
[31m-[m
[31m-# A=dj.U('animal_id','session').aggr(meso.ScanDone*meso.StackCoordinates.UnitInfo,maxz='max(stack_z)')[m
[31m-# A & 'maxz>500'[m
[31m-select s.animal_id, s.session, s.scan_idx, max(info.stack_z) from __scan_done s,  __stack_coordinates__unit_info  info[m
[31m-where s.animal_id=info.animal_id and s.session=info.session[m
[31m-group by info.animal_id,info.session,info.scan_idx having max(info.stack_z)>500;[m
[31m-[m
[31m-##############################################[m
[31m-# loading spontaneous data info[m
[31m-##################################333[m
[31m-[m
[31m-A=dj.U('animal_id','session','scan_idx').aggr(meso.ScanDone*meso.StackCoordinates.UnitInfo,maxz='max(stack_z)')[m
[31m-keyinfo = (dj.U('animal_id','session','scan_idx')&(A & 'maxz>450')).fetch()[m
[31m-[m
[31m-[m
[31m-Spon = pd.DataFrame()[m
[31m-[m
[31m-for aid,ses,iscan in keyinfo:[m
[31m-    [m
[31m-    key = {'animal_id': aid, 'session':ses, 'scan_idx':iscan}[m
[31m-        [m
[31m-    flip_times = (stimulus.Trial & key).fetch('flip_times')[m
[31m-    last_flip_times = flip_times[-1].squeeze()[-1][m
[31m-    [m
[31m-    if len(stimulus.Sync & key)>0:[m
[31m-        frame_times = (stimulus.Sync & key).fetch1('frame_times').squeeze() [m
[31m-        slice_num = len(np.unique((meso.ScanInfo.Field & key).fetch('z')))[m
[31m-        field_offset=0[m
[31m-        frame_times = frame_times[field_offset::slice_num][m
[31m-        [m
[31m-        spon_frame_start = np.where(frame_times>last_flip_times)[0][0][m
[31m-        spon_dur = len(frame_times)-spon_frame_start[m
[31m-        b= {'spon_frame_start':spon_frame_start, 'spon_framedur':spon_dur}[m
[31m-        out_= {**key, **b}[m
[31m-        print(out_) [m
[31m-        Spon= Spon.append(out_,ignore_index=True)[m
[31m-cols=['animal_id','session','scan_idx','spon_frame_start','spon_framedur'][m
[31m-Spon = Spon[cols][m
[31m-[m
[31m-stiminfo = pd.read_csv('./doc/meso_stim_perscan.csv')[m
[31m-out = stiminfo.merge(Spon)[m
[31m-out.to_csv('./doc/meso_stim_spon_perscan.csv') [m
[31m-[m
[31m-[m
[31m-[m
[31m-[m
[31m-[m
[31m-[m
[31m-[m
[31m-# case 1 [m
[31m-###### stimulus.Trial &'animal_id=17797,session=6, t.scan_idx=6' * stimulus.Condtion[m
[31m-# select t.animal_id, t.session, t.scan_idx, t.trial_idx, t.condition_hash, c.stimulus_type [m
[31m-# from trial  t [m
[31m-# left join pipeline_stimulus.condition c [m
[31m-# on t.condition_hash = c.condition_hash[m
[31m-# where t.animal_id=17797 and t.session=6 and t.scan_idx=6 ;[m
[31m-[m
[31m-[m
[31m-[m
[31m-# case 2:[m
[31m-# stimulus.Trial * stimulus.Condtion[m
[31m-# will result valid number on 'animal_id=17797,session=6, t.scan_idx=6' [m
[31m-#########################################3[m
[31m-# select t.animal_id, t.session, t.scan_idx, t.trial_idx, t.condition_hash, c.stimulus_type [m
[31m-# from trial  t [m
[31m-# left join pipeline_stimulus.condition c [m
[31m-# on t.condition_hash = c.condition_hash[m
[31m-# and t.animal_id=17797 and t.session=6 and t.scan_idx=6 ;[m
[1mdiff --git a/Spon/pipeline_config.json b/Spon/pipeline_config.json[m
[1mdeleted file mode 100755[m
[1mindex d3e05e0..0000000[m
[1m--- a/Spon/pipeline_config.json[m
[1m+++ /dev/null[m
[36m@@ -1,4 +0,0 @@[m
[31m-{[m
[31m-    "path.mounts": "/mnt/",[m
[31m-    "display.tracking": false[m
[31m-}[m
\ No newline at end of file[m
[1mdiff --git a/Spon/spon.py b/Spon/spon.py[m
[1mdeleted file mode 100755[m
[1mindex 1becd8c..0000000[m
[1m--- a/Spon/spon.py[m
[1m+++ /dev/null[m
[36m@@ -1,517 +0,0 @@[m
[31m-#!/usr/bin/env python3[m
[31m-# -*- coding: utf-8 -*-[m
[31m-"""[m
[31m-Created on Sat Sep 26 11:36:38 2020[m
[31m-[m
[31m-@author: sang[m
[31m-"""[m
[31m-[m
[31m-#[m
[31m-%run access_db.py[m
[31m-#runfile('access_db.py')[m
[31m-[m
[31m-import datajoint as dj[m
[31m-import numpy as np[m
[31m-import pandas as pd[m
[31m-import matplotlib.pyplot as plt[m
[31m-from pipeline import experiment, reso, meso, fuse, stack,  treadmill, pupil, shared[m
[31m-from stimulus import stimulus[m
[31m-anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')  [m
[31m-[m
[31m-#experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')[m
[31m-#meso = dj.create_virtual_module('meso', 'pipeline_meso')[m
[31m-#pupil = dj.create_virtual_module('pupil', 'pipeline_eye')[m
[31m-#stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')[m
[31m-#treadmill = dj.create_virtual_module('treadmill', 'pipeline_treadmill')[m
[31m-# shared = dj.create_virtual_module('shared', 'pipeline_shared')[m
[31m-[m
[31m-[m
[31m-[m
[31m-# schema = dj.schema('sang_neuro', locals(), create_tables=True)[m
[31m-schema = dj.schema('sang_neuro')[m
[31m-[m
[31m-@schema[m
[31m-class Sponscan(dj.Computed):[m
[31m-    definition = """ # scanlist with spontaneous recording[m
[31m-    [m
[31m-    -> meso.ScanInfo[m
[31m-    ---[m
[31m-    spon_frame_start                  : int unsigned      #frame start index[m
[31m-    spon_frame_dur                    : int unsigned      #frame duration[m
[31m-    caclulation_time=CURRENT_TIMESTAMP    : timestamp     # automatic[m
[31m-    """[m
[31m-    [m
[31m-    [m
[31m-    def make(self, key):[m
[31m-        [m
[31m-        [m
[31m-        stim_key = stimulus.Sync & key[m
[31m-        flip_times = (stimulus.Trial & key).fetch('flip_times')[m
[31m-        if len(stim_key)>0 and len(flip_times)>0:[m
[31m-            [m
[31m-            last_flip_times = np.nanmax(flip_times[-1].flatten())[m
[31m-            [m
[31m-            frame_times = stim_key.fetch1('frame_times').squeeze() [m
[31m-            slice_num = len(np.unique((meso.ScanInfo.Field & key).fetch('z')))[m
[31m-            field_offset=0[m
[31m-            if slice_num>0:[m
[31m-                frame_times = frame_times[field_offset::slice_num][m
[31m-            [m
[31m-                [m
[31m-            print(key)[m
[31m-            spon_frames = np.where(frame_times>last_flip_times)[m
[31m-            if len(spon_frames)>0 and len(spon_frames[0])>0:[m
[31m-                [m
[31m-                # print(spon_frames[0].shape)[m
[31m-                spon_frame_start = spon_frames[0][0][m
[31m-                spon_dur = len(frame_times)-spon_frame_start[m
[31m-                [m
[31m-                out_= key.copy()[m
[31m-                out_['spon_frame_start'] =spon_frame_start[m
[31m-                out_['spon_frame_dur'] = spon_dur[m
[31m-                [m
[31m-                [m
[31m-                print(out_)[m
[31m-    [m
[31m-                [m
[31m-                self.insert1(out_)[m
[31m-                [m
[31m-        [m
[31m-# Sponscan.populate()        [m
[31m-[m
[31m-#animal_ids: 17797, 17977,18142, 18252[m
[31m-[m
[31m-    [m
[31m-@schema[m
[31m-class SponScanSel(dj.Computed):[m
[31m-    definition = """ # scan depth summary[m
[31m-    [m
[31m-    -> Sponscan[m
[31m-    depth_thr           : float         # max depth threshold for scan selection[m
[31m-    depth_interval_thr  : float         # max depth interval threshold for scan selection[m
[31m-    spon_framedur_thr   : int unsigned  # threshold for spontaneous frame duration [m
[31m-    ---[m
[31m-    spon_frame_dur                    : int unsigned      #frame duration[m
[31m-    ndepth                            : int unsigned      # number of field depths[m
[31m-    field_depth                       : blob              #field depth list[m
[31m-    caclulation_time=CURRENT_TIMESTAMP    : timestamp     # automatic[m
[31m-    """[m
[31m-    def make(self, key):[m
[31m-        import numpy as np[m
[31m-        key1 = key.copy()[m
[31m-        key1['depth_thr'] = 450[m
[31m-        key1['depth_interval_thr'] = 100[m
[31m-        key1['spon_framedur_thr'] = 3000[m
[31m-        [m
[31m-        [m
[31m-        # field depth, this is not an accurate depth for individual cells[m
[31m-        depths = (meso.ScanInfo.Field & key).fetch('z')[m
[31m-        maxz = np.nan[m
[31m-        max_depth_interval = np.nan[m
[31m-        [m
[31m-        [m
[31m-        if len(depths)>1:            [m
[31m-            print(key)[m
[31m-            maxz = depths.max()[m
[31m-            udepth = np.unique(depths)[m
[31m-            if len(udepth)>1:[m
[31m-                max_depth_interval = np.max(np.diff(udepth))[m
[31m-        [m
[31m-        spon_frame_dur = (Sponscan() &key).fetch1('spon_frame_dur')[m
[31m-        [m
[31m-        if maxz > key1['depth_thr'] and max_depth_interval > key1['depth_interval_thr'] and spon_frame_dur> key1['spon_framedur_thr']:[m
[31m-            [m
[31m-            key1['ndepth'] = len(udepth)[m
[31m-            key1['field_depth'] = udepth[m
[31m-            key1['spon_frame_dur'] = spon_frame_dur[m
[31m-            self.insert1(key1)[m
[31m-            [m
[31m-# *animal_id    *session    *scan_idx    *pipe_version  *segmentation_ *unit_id    brain_area             [m
[31m-@schema            [m
[31m-class AreaMembership(dj.Computed):[m
[31m-    definition = """ # this is a replicate of anatomy.AreaMembership to populate mylist[m
[31m-    -> meso.ScanInfo[m
[31m-    -> shared.SegmentationMethod[m
[31m-    ---    [m
[31m-    caclulation_time=CURRENT_TIMESTAMP    : timestamp     # automatic    [m
[31m-    """[m
[31m-    @property[m
[31m-    def key_source(self):[m
[31m-        return (meso.ScanInfo*shared.SegmentationMethod) & anatomy.AreaMask & meso.ScanDone& {'pipe_version':1, 'segmentation_method':6} [m
[31m-        #return key1 #& {'animal_id': 17797, 'session': 6, 'scan_idx': 4}[m
[31m-    [m
[31m-    class Unit(dj.Part):[m
[31m-        definition = """ [m
[31m-        -> master      [m
[31m-        unit_id                               : int          # unit id[m
[31m-        ---[m
[31m-        -> anatomy.Area        [m
[31m-        """[m
[31m-[m
[31m-    def make(self, key):[m
[31m-        fields = (meso.ScanInfo.Field & anatomy.AreaMask &(meso.ScanDone& key)).fetch('field')[m
[31m-        [m
[31m-        #print(key)[m
[31m-        #print(len(fields))[m
[31m-        if len(fields)>0:                   [m
[31m-            self.insert1(key)[m
[31m-            [m
[31m-        for field_id in fields:[m
[31m-            field_key = key.copy()[m
[31m-            field_key['field'] = field_id[m
[31m-            area_masks, areas =(anatomy.AreaMask & field_key).fetch('mask','brain_area')[m
[31m-            [m
[31m-            area_mask = np.nan*np.ones(area_masks[0].shape)[m
[31m-            for iarea in range(len(area_masks)):[m
[31m-                area_mask[area_masks[iarea]>0] = iarea[m
[31m-            [m
[31m-            [m
[31m-            #units selected from a specific field[m
[31m-            unit_ids, px_x, px_y = (meso.ScanSet.UnitInfo & [m
[31m-                                    (meso.ScanSet.Unit& field_key)).fetch('unit_id','px_x','px_y')[m
[31m-                       [m
[31m-            dj.conn()[m
[31m-            for i, uid in enumerate(unit_ids):                [m
[31m-                area_idx = (area_mask[round(px_y[i]),round(px_x[i])])[m
[31m-                tup_out = key.copy()            [m
[31m-                tup_out['unit_id'] = uid                [m
[31m-                if np.isnan(area_idx):[m
[31m-                    tup_out['brain_area']= 'unknown'                    [m
[31m-                else:                    [m
[31m-                    tup_out['brain_area']= areas[int(area_idx)]                [m
[31m-                AreaMembership.Unit.insert1(tup_out)[m
[31m-            [m
[31m-            [m
[31m-#popout = AreaMembership.populate(reserve_jobs=True,display_progress=True,[m
[31m-#                                 suppress_errors=True,[m
[31m-#                                 return_exception_objects=True,[m
[31m-#                                 order="random")           [m
[31m-[m
[31m-          [m
[31m-def exclude_border_neighbor(unit_loc_pix, border_loc_pix, field_shape, microns_per_pixel=(1,1),[m
[31m-                           px_dist_edge=10, dist_border=30):[m
[31m-    """[m
[31m-    remove units close to field edge and area borders[m
[31m-    [m
[31m-    arguments:[m
[31m-    unit_loc_pix : tuple of lists of y and x in pixel location of units[m
[31m-    border_loc_pix : tuple of lists of y and x in pixel location of area border[m
[31m-                    if no border is given within a field,  ([],[])[m
[31m-    field_shape : tupe of y- and x- size[m
[31m-    px_dist_edge : pixel distance to field edge[m
[31m-    dist_border : distance(um) threshold to area border [m
[31m-    [m
[31m-    return:[m
[31m-        cidx[0]   : select unit ids[m
[31m-        cmask     : mask to display units with borders[m
[31m-        [m
[31m-    """[m
[31m-    import numpy as np[m
[31m-    px_y, px_x = unit_loc_pix[m
[31m-    [m
[31m-    # edge mask[m
[31m-    mask = np.ones(field_shape)*True[m
[31m-    mask[slice(px_dist_edge),:]=False[m
[31m-    mask[slice(field_shape[0]-px_dist_edge, field_shape[0]),:]=False[m
[31m-    mask[:,slice(px_dist_edge)]=False[m
[31m-    mask[:,slice(field_shape[1]-px_dist_edge, field_shape[1])]=False[m
[31m-    #return mask[m
[31m-    if len(border_loc_pix[0])>0:[m
[31m-[m
[31m-        df_y=  px_y[:,np.newaxis] -  border_loc_pix[0][m
[31m-        df_x=  px_x[:,np.newaxis] -  border_loc_pix[1]    [m
[31m-        [m
[31m-        dist = np.sqrt((df_y*microns_per_pixel[0])**2+(df_x*microns_per_pixel[1])**2)[m
[31m-        # minium distance to any area border[m
[31m-        mindist = np.min(dist,axis=1)[m
[31m-        cidx  = np.where(np.logical_and(mindist>dist_border, mask[px_y, px_x]))[m
[31m-    else: # when bodrder_loc_pix is empty[m
[31m-        cidx  = np.where(mask[px_y, px_x])[m
[31m-        [m
[31m-    cmask = np.zeros(field_shape)    [m
[31m-    cmask[border_loc_pix[0],border_loc_pix[1]] = 1    [m
[31m-    cmask[px_y[cidx[0]], px_x[cidx[0]] ]=2[m
[31m-    [m
[31m-    return cidx[0], cmask[m
[31m-[m
[31m-[m
[31m-def remove_overlap_field(field_center, field_size, field_ids):[m
[31m-    """ remove overlap area between fields, always overlap removed in the later field number[m
[31m-    [m
[31m-    field_center: x, y, z, tuple of field_list[m
[31m-    field_size: height, width, tuple of field_list[m
[31m-    field_ids: list of field id[m
[31m-    [m
[31m-    return:[m
[31m-        select_range:    list of field_range [[y0,x0], [y1, x1],field_order] [m
[31m-        field_ids:       list of field ids, i.e., field_ids[field_order][m
[31m-    """[m
[31m-    import numpy as np[m
[31m-    [m
[31m-    x, y, z = field_center[m
[31m-    height, width = field_size [m
[31m-    [m
[31m-    select_range = [][m
[31m-    fz =[][m
[31m-    for fid in range(len(x)):[m
[31m-        y0 = round(y[fid] - height[fid]/2)[m
[31m-        y1 = round(y[fid] + height[fid]/2)[m
[31m-        x0 = round(x[fid] - width[fid]/2)[m
[31m-        x1 = round(x[fid] + width[fid]/2)[m
[31m-        [m
[31m-        lt = np.array([y0,x0]) # left-top corner[m
[31m-        rb = np.array([y1,x1]) #right-bottom corner[m
[31m-        if z[fid] in fz:[m
[31m-            ix = np.where(fz==z[fid])[0][0][m
[31m-            for j in range(len(select_range[ix])):[m
[31m-                rb_ = select_range[ix][j][1][m
[31m-                iou = rb_ -lt[m
[31m-                if np.all(iou>0): # if two fields are overlap[m
[31m-                    [m
[31m-                    # find directions of iou are smaller than field size[m
[31m-                    overlap_direction = np.where (([height[fid], width[fid]] -iou)>0)[0] [m
[31m-                    lt[overlap_direction] = rb_[overlap_direction][m
[31m-                    [m
[31m-            select_range[ix].append([lt, rb, fid])[m
[31m-        else:[m
[31m-            fz.append(z[fid])[m
[31m-            select_range.append([[lt, rb, fid]])[m
[31m-            [m
[31m-    out = []    [m
[31m-    for i in range(len(select_range)):[m
[31m-        out += select_range[i][m
[31m-        [m
[31m-    flist=[i[2] for i in out][m
[31m-    [m
[31m-    return out, field_ids[flist][m
[31m-[m
[31m-@schema[m
[31m-class BorderRestrict(dj.Computed):[m
[31m-    definition=""" # remove units close to AreaBorder and field edge[m
[31m-    -> AreaMembership    [m
[31m-    edge_distance_px                      : smallint    # pixel distance to field edge[m
[31m-    border_distance_um                    : float  # distance (um) to Areaborder[m
[31m-    field_overlap_remove                        : tinyint   # boolean to decide whether overalp field is removed[m
[31m-    --- [m
[31m-    caclulation_time=CURRENT_TIMESTAMP    : timestamp     # automatic  [m
[31m-    """[m
[31m-    [m
[31m-    class Unit(dj.Part):[m
[31m-        definition = """ [m
[31m-        -> master      [m
[31m-        unit_id                               : int          # unit id[m
[31m-        ---    [m
[31m-        um_x                : smallint      # x-coordinate of centroid in motor coordinate system[m
[31m-        um_y                : smallint      # y-coordinate of centroid in motor coordinate system[m
[31m-        um_z                : smallint      # z-coordinate of mask relative to surface of the cortex[m
[31m-        """[m
[31m-    class FieldMask(dj.Part):[m
[31m-        definition = """[m
[31m-        ->master[m
[31m-        field                              : int          # field id[m
[31m-        ---[m
[31m-        unit_image                         : blob          # area border and unit[m
[31m-        """[m
[31m-        [m
[31m-    def make(self, key):[m
[31m-        [m
[31m-        key1 = key.copy()[m
[31m-        key1['edge_distance_px'] = 10        [m
[31m-        key1['field_overlap_remove'] = 1[m
[31m-        [m
[31m-        [m
[31m-        field_ids, um_height, um_width,fx0, fy0, fz0 = (meso.ScanInfo.Field&key).fetch([m
[31m-                'field', 'um_height', 'um_width', 'x', 'y', 'z')        [m
[31m-        [m
[31m-        if key1['field_overlap_remove']>0:[m
[31m-            field_center = (fx0, fy0, fz0)[m
[31m-            field_size = (um_height, um_width)[m
[31m-            field_restriction, field_order = remove_overlap_field(field_center, field_size, field_ids)[m
[31m-        [m
[31m-        for bdist in [30, 50, 70, 100]:[m
[31m-        [m
[31m-            key1['border_distance_um'] = bdist[m
[31m-            self.insert1(key1)[m
[31m-            print(key1)[m
[31m-            for ifid, fid in enumerate(field_ids):                [m
[31m-                field_key = key1.copy()[m
[31m-                field_key['field']=fid[m
[31m-                [m
[31m-                 #units from a specific field[m
[31m-                unit_ids, px_x, px_y, um_x, um_y = (meso.ScanSet.UnitInfo &[m
[31m-                                        (meso.ScanSet.Unit& field_key)).fetch('unit_id','px_x','px_y', 'um_x','um_y')[m
[31m-                [m
[31m-                [m
[31m-                # create area mask[m
[31m-                area_masks, areas =(anatomy.AreaMask & field_key).fetch('mask','brain_area')[m
[31m-                area_mask = np.zeros(area_masks[0].shape)[m
[31m-                [m
[31m-                for iarea in range(len(area_masks)):                    [m
[31m-                    area_mask[area_masks[iarea]>0] = iarea+1[m
[31m-                    [m
[31m-                import cv2[m
[31m-                edge_image = cv2.Laplacian(area_mask,cv2.CV_64F)[m
[31m-                border_loc_pix  = np.where(abs(edge_image)>0) [m
[31m-                [m
[31m-                if len(border_loc_pix)<2 or len(border_loc_pix[0])==0: # when single area is imaged[m
[31m-                    border_loc_pix = ([],[])                       [m
[31m-                [m
[31m-                microns_per_pixel = (meso.ScanInfo.Field& field_key).microns_per_pixel            [m
[31m-                uix, mask_img = exclude_border_neighbor((px_y,px_x), border_loc_pix, [m
[31m-                                                       area_mask.shape, [m
[31m-                                                       microns_per_pixel,[m
[31m-                                                       field_key['edge_distance_px'], [m
[31m-                                                       field_key['border_distance_um'])[m
[31m-               [m
[31m-                [m
[31m-                unit_ids_sel= unit_ids[uix][m
[31m-                [m
[31m-                if key1['field_overlap_remove']>0:                    [m
[31m-                    xs = um_x[uix][m
[31m-                    ys = um_y[uix][m
[31m-                    inx = int(np.where(field_order==fid)[0])[m
[31m-                    frestrict = field_restriction[inx][m
[31m-                    conds = np.vstack((frestrict[0][0]<ys, ys<frestrict[1][0], frestrict[0][1]<xs, xs<frestrict[1][1]))[m
[31m-                    uix2 = np.where( conds.all(axis=0))[0]                [m
[31m-                    unit_ids_sel = unit_ids_sel[uix2][m
[31m-                [m
[31m-                [m
[31m-                [m
[31m-                field_key['unit_image']= mask_img.astype('int8')[m
[31m-                dj.conn()[m
[31m-                BorderRestrict.FieldMask.insert1(field_key)[m
[31m-                [m
[31m-                dj.conn()[m
[31m-                for iu in unit_ids_sel:[m
[31m-                    unit_key = key1.copy()[m
[31m-                    unit_key['unit_id']= iu[m
[31m-                    ix  = np.where(unit_ids==iu)[0][m
[31m-                    unit_key['um_z'] = int(fz0[ifid])[m
[31m-                    unit_key['um_x'] = um_x[ix][0][m
[31m-                    unit_key['um_y'] = um_y[ix][0][m
[31m-                    BorderRestrict.Unit.insert1(unit_key)[m
[31m-                print(unit_key)[m
[31m-                [m
[31m-#popout = BorderRestrict.populate(order="random",display_progress=True,suppress_errors=True)[m
[31m-#[m
[31m-[m
[31m-@schema            [m
[31m-class LayerMembership(dj.Computed):[m
[31m-    definition = """ # this is a replicate of anatomy.AreaMembership to populate mylist[m
[31m-    -> meso.ScanInfo[m
[31m-    -> shared.SegmentationMethod[m
[31m-    ---    [m
[31m-    caclulation_time=CURRENT_TIMESTAMP    : timestamp     # automatic    [m
[31m-    """[m
[31m-    [m
[31m-    class Unit(dj.Part):[m
[31m-        definition = """ [m
[31m-        -> master      [m
[31m-        unit_id                               : int          # unit id[m
[31m-        ---[m
[31m-        -> anatomy.Layer        [m
[31m-        """[m
[31m-    @property[m
[31m-    def key_source(self):[m
[31m-        return (meso.ScanInfo*shared.SegmentationMethod) & meso.ScanDone& {'pipe_version':1, 'segmentation_method':6} [m
[31m-[m
[31m-    def make(self, key):[m
[31m-        field_keys = ((meso.ScanInfo.Field*shared.SegmentationMethod) &(meso.ScanDone& key)).fetch('KEY')[m
[31m-        self.insert1(key)[m
[31m-        [m
[31m-        layers, zstart, zend = anatomy.Layer.fetch('layer','z_start','z_end')[m
[31m-            [m
[31m-        for field_key in field_keys:[m
[31m-            depth = (meso.ScanInfo.Field &field_key).fetch('z')[m
[31m-             [m
[31m-            mask_keys = (meso.ScanSet.Unit&field_key).fetch('KEY')[m
[31m-            [m
[31m-            idx = np.all(np.vstack((zstart<depth[0], zend>=depth[0])),0)[m
[31m-            if len(np.where(idx))==1:[m
[31m-                layer = layers[idx][0][m
[31m-            else:[m
[31m-                layer = 'unset'[m
[31m-                [m
[31m-            for i in range(len(mask_keys)):[m
[31m-                mask_keys[i]['layer']=layer[m
[31m-            dj.conn()[m
[31m-            [m
[31m-            LayerMembership.Unit.insert(mask_keys)[m
[31m-[m
[31m-#popout = LayerMembership.populate(display_progress=True)[m
[31m-[m
[31m-@schema[m
[31m-class SpontaneousActivity(dj.Computed):[m
[31m-    definition = """[m
[31m-    -> BorderRestrict[m
[31m-    -> shared.SpikeMethod[m
[31m-    -> anatomy.Area[m
[31m-    -> anatomy.Layer[m
[31m-    ---  [m
[31m-    unit_number                      : int           # number of units[m
[31m-    unit_ids                         : blob           # list of unit ids[m
[31m-    mean_activity                    :  external-deeplab   # timesamples [m
[31m-    activity_matrix                  :  external-deeplab   # timesamples x number of units[m
[31m-    spa_ctime = CURRENT_TIMESTAMP    : timestamp     # automatic    [m
[31m-    """[m
[31m-    @property[m
[31m-    def key_source(self):[m
[31m-        return BorderRestrict&LayerMembership.proj()&SponScanSel.proj()[m
[31m-    [m
[31m-    [m
[31m-    def make(self, key):[m
[31m-        units = (AreaMembership.Unit*LayerMembership.Unit)&BorderRestrict.Unit&key[m
[31m-        a = (dj.U('brain_area','layer')&units).fetch()[m
[31m-        brain_areas, layers = zip(*a)[m
[31m-        n = len(a)[m
[31m-        [m
[31m-        for i in range(n):[m
[31m-            ba = brain_areas[i][m
[31m-            layer = layers[i]            [m
[31m-            outkey = key.copy()[m
[31m-            [m
[31m-            area_keys = units&{'brain_area':ba,'layer':layer}[m
[31m-            outkey['brain_area'] = ba[m
[31m-            outkey['layer'] = layer[m
[31m-            outkey['unit_number'] = len(area_keys)[m
[31m-            outkey['spike_method']=5[m
[31m-            outkey['unit_ids'] = area_keys.fetch('unit_id')[m
[31m-            Trace = (meso.Activity.Trace&area_keys & {'spike_method':5}).fetch('trace')[m
[31m-            spon_start_idx = (Sponscan&key).fetch('spon_frame_start')[0][m
[31m-            m = np.vstack(Trace).T[m
[31m-            m = m[spon_start_idx:,:] [m
[31m-            outkey['activity_matrix'] = m        [m
[31m-            mactivity = np.mean(m,1)[m
[31m-            outkey['mean_activity'] = mactivity    [m
[31m-            dj.conn()[m
[31m-            self.insert1(outkey)[m
[31m-[m
[31m-popout = SpontaneousActivity.populate(order="random",display_progress=True,suppress_errors=True)[m
[31m-[m
[31m-[m
[31m-#key = {'animal_id': 17795, 'session': 5, 'scan_idx': 5, 'pipe_version': 1, 'segmentation_method': 6}[m
[31m-#rel = meso.Activity.Trace&(BorderRestrict.Unit&key&'border_distance_um =100')[m
[31m-#[m
[31m-#[m
[31m-## check number of unit in each brain area[m
[31m-#rel =meso.ScanSet.Unit* (AreaMembership.Unit&(BorderRestrict.Unit&key&'border_distance_um =100'))[m
[31m-#dj.U('field','brain_area').aggr(rel, n='count(brain_area)')[m
[31m-#[m
[31m-#[m
[31m-##aunit_id, brain_area = (AreaMembership.Unit & key).fetch('unit_id','brain_area')[m
[31m-##funit_id, trace = (meso.Activity.Trace&key).fetch('unit_id','trace')[m
[31m-[m
[31m-## SponScanSel.populate()            [m
[31m-#key = (meso.ScanDone&SponScanSel&anatomy.AreaMembership).fetch('KEY')[0] [m
[31m-#[m
[31m-#[m
[31m-#key1 =  {'animal_id': 17358, 'session': 1, 'scan_idx': 13, 'pipe_version': 1, 'field': 1}[m
[31m-#key1 =  {'animal_id': 17358, 'session': 1, 'scan_idx': 13, 'pipe_version': 1, 'segmentation_method': 6}[m
[31m-#[m
[31m-#experiment.Scan * experiment.Session & key1[m
[31m-#field_key = ((fuse.ScanSet& anatomy.AreaMask) & key1).fetch('KEY')[0][m
[31m-#area_masks, areas = (anatomy.AreaMask & field_key).fetch('mask','brain_area')[m
[31m-#[m
[31m-#[m
[31m-#[m
[31m-#meso.ScanSet.UnitInfo &field_key[m
\ No newline at end of file[m
[1mdiff --git a/commons b/commons[m
[1m--- a/commons[m
[1m+++ b/commons[m
[36m@@ -1 +1 @@[m
[31m-Subproject commit 5c6d91699403043adbc635e005a3172446a209c9[m
[32m+[m[32mSubproject commit 5c6d91699403043adbc635e005a3172446a209c9-dirty[m
[1mdiff --git a/pipeline b/pipeline[m
[1m--- a/pipeline[m
[1m+++ b/pipeline[m
[36m@@ -1 +1 @@[m
[31m-Subproject commit 8f23e1e57fc1fe182e0a3270a258a3c2552fa45c[m
[32m+[m[32mSubproject commit 8f23e1e57fc1fe182e0a3270a258a3c2552fa45c-dirty[m
[1mdiff --git a/stimuli b/stimuli[m
[1m--- a/stimuli[m
[1m+++ b/stimuli[m
[36m@@ -1 +1 @@[m
[31m-Subproject commit e2cce9ceab74ec37446839a46b583a34b27fe6fe[m
[32m+[m[32mSubproject commit e2cce9ceab74ec37446839a46b583a34b27fe6fe-dirty[m
[1mdiff --git a/stimulus-pipeline b/stimulus-pipeline[m
[1m--- a/stimulus-pipeline[m
[1m+++ b/stimulus-pipeline[m
[36m@@ -1 +1 @@[m
[31m-Subproject commit 36d4cfd342e74d76190c50c5e089f87e57bd0790[m
[32m+[m[32mSubproject commit 36d4cfd342e74d76190c50c5e089f87e57bd0790-dirty[m
