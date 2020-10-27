pd.DataFrame(dj.U('animal_id') & stack.Area)
    
    animal_id
0       18142     good for inter-area/lamina, spon, stim repetition
1       20892    --> multiple brain area, only L2/3 and L4 for some scan and not repetition
2       20897     good for inter-area/lamina, spon, stim repetition
3       21067      good for inter-area/lamina, stim repetition  not for spontaneous
4       21154    ---> single layer image
5       21553    ---> single layer image
6       21844       ---> single layer image only L2/3
7       21900    ---> single layer image
8       22083    ---> single layer image
9       22279    ---> single layer image
10      23224       ---> single layer image only L2/3
11      23343    ---> single layer image


aid = 'animal_id = 20897'
t = (stimulus.Trial & aid )*stimulus.Condition

out = dj.U('session','scan_idx','condition_hash','stimulus_type').aggr(t,ncount='count(stimulus_type)')

dj.U('stimulus_type') &(out & 'ncount>9')
# Out[381]: 
# *stimulus_type
# +------------+
# stimulus.Fancy
# stimulus.Singl
# stimulus.Clip 
# stimulus.Frame
#  (4 tuples)
 
t= dj.U('animal_id','session','scan_idx').aggr(meso.StackCoordinates.UnitInfo & aid, depth='max(stack_z)')
 
#  Out[396]: 
# *session    *scan_idx    depth         
# +---------+ +----------+ +------------+
# 3           11           234.6106414794
# 3           14           243.9084014892
# 4           11           239.9379577636
# 4           16           236.7787933349
# 5           18           240.8100891113
# 5           29           240.2297363281
# 6           17           239.7984619140
#    ...
#  (22 tuples)


keys  = (t & 'depth>350').fetch('KEY')
ikey =1
meso.ScanInfo.Field & keys[ikey]  # depth info

#stimulus type with repetition
t = (stimulus.Trial & keys[ikey] )*stimulus.Condition
out = dj.U('session','scan_idx','condition_hash','stimulus_type').aggr(t,ncount='count(stimulus_type)')

out &(dj.U('stimulus_type') &(out & 'ncount>9'))




keySource = ((experiment.Scan) *  (shared.SegmentationMethod) *  (shared.PipelineVersion) & fuse.ScanDone) &anatomy.AreaMask

out = keySource.fetch('KEY')
aid = 'animal_id=18142'
anatomy.AreaMask&aid

out1 = (anatomy.AreaMask&aid).fetch('KEY')



aid = 'animal_id=18252'
keySource  & 'animal_id=18252 and pipe_version = 1'


anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')

alist = pd.DataFrame(dj.U('animal_id') & (experiment.Scan *anatomy.Area * shared.Field) & 'animal_id = 18142'


alist.index(alist['animal_id']==18142).to_list()

