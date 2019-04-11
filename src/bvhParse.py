import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *
import src.pathDefine

# bvh_file = 'chacha.bvh'
bvh_path = src.pathDefine.raw_bvh_file
bvh_feature_path = src.pathDefine.features_bvh_file
header_path = src.pathDefine.header_output


p = BVHParser()
data_all = [p.parse(bvh_path)]
print_skel(data_all[0])
dr_pipe = Pipeline([
    ('param', MocapParameterizer('quat')),
])
quat = dr_pipe.fit_transform(data_all)

# print(data_all[0].values.shape)

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# print(xx[0].values)
# print(type(xx[0].values))
# print(xx[0].values.shape)
# quat[0].values.to_csv("bvh_feature.csv")
print(quat[0].values.shape)
print(quat[0].values[0])
# quat[0].values.to_csv(bvh_feature_path)




# f = open(bvh_path,'r')
# result = list()
# for line in open(bvh_path):
#     line = f.readline()
#     result.append(line)
#     if 'Frame Time' in line:
#         break
# f.close()
# open(header_path, 'w').write('%s' % ''.join(result))
