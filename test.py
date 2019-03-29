import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle

import os
import sys
import math
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *
p = BVHParser()

data_all = [p.parse('/Users/mac/Documents/Mocap/Example1.bvh')]
print_skel(data_all[0])

from pymo.parsers import BVHParser
import pymo.viz_tools

parser = BVHParser()

parsed_data = parser.parse('/Users/mac/Documents/Mocap/Example1.bvh')
mp = MocapParameterizer('quat')

dr_pipe = Pipeline([
    ('param', MocapParameterizer('quat')),
])
xx = dr_pipe.fit_transform(data_all)

print(data_all[0].values.shape)
# print(data_all[0].values)

# #显示所有列
# xx[0].values.set_option('display.max_columns', None)
# #显示所有行
# xx[0].values.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
#pd.set_option('max_colwidth',100)


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
print(xx[0].values)
print(type(xx[0].values))
print(xx[0].values.shape)

# quat = [ -0.0117277, -0.9825456, -0.1313857, 0.1311659 ]
# quat = [0.046736,-0.981503,0.123351,0.138749]
# quat = [-0.207485,0.000000,0.000000,0.978238 ]
# quat = [ -0.011728, -0.982546, -0.131386, 0.131166 ]
# quat = [-0.271,0.653,0.271,0.653]
quat = [ 0.2705981, -0.6532815, 0.2705981, 0.6532815]
x = quat[0]
y = quat[1]
z = quat[2]
w = quat[3]
R = np.asarray([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
                ])
Epsilon = 0.00000001;
Threshold = 1.0 - Epsilon;

r11 = -2 * (x * y - w * z)
r12 = w * w - x * x + y * y - z * z
r21 = 2 * (y * z + w * x)
r31 = -2 * (x * z - w * y)
r32 = w * w - x * x - y * y + z * z


if r21 < -Threshold or r21 > Threshold:
    sign = np.sign(r21)
    zz = -2 * sign * math.atan2(x,w)
    xx = 0
    yy = sign * (math.pi/2.0)
else:
    zz= math.atan2(r11,r12)
    xx = math.asin(r21)
    yy = math.atan2(r31,r32)

# r==y p==x
angleR = yy*180/math.pi
angleP = xx*180/math.pi
angleY = zz*180/math.pi

test = [zz,xx,yy]
test2 = [angleY,angleP,angleR]
print(test)
print(test2)