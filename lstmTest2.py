import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
mpl.rcParams['legend.fontsize']=10#图例字号
fig = plt.figure()
ax = fig.gca(projection='3d')#三维图形
theta = np.linspace(-4* np.pi,4* np.pi,100)
z = np.linspace(-4,4,100)*0.3#测试数据
r = z**3+1
x = r * np.sin(theta)
y = r * np.cos(theta)
# plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')

# ax.plot(x, y, z, label='parametric curve')
ax.plot(x, y, z, 'r', -x, -y, z, 'b--')
# 可选参数[fmt] 是一个字符串来定义图的基本属性如：颜色（color），点型（marker），线型（linestyle），
# 具体形式  fmt = '[color][marker][line]'
# fmt接收的是每个属性的单个字母缩写，例如：
# plot(x, y, 'bo-')  # 蓝色圆点实线
# 若属性用的是全名则不能用*fmt*参数来组合赋值，应该用关键字参数对单个属性赋值如：
# plot(x,y2,color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
# plot(x,y3,color='#900302',marker='+',linestyle='-')

ax.legend()
plt.show()