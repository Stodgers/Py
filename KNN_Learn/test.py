from numpy import *
from os import listdir
import operator
a = tile([1, 2, 3], (1, 1))
ashape = a.shape[0]
print(a)
print(ashape)
aa = zeros((1,ashape))
dd = tile(a,(1,1))-aa
dd = dd**2
ss = dd.sum(axis=1)
ss = ss**0.5
print(dd)
print(ss)
print('------------')
cc = {}

