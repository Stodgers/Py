import pandas as pd
import numpy as np
'''
d = {'a' : 10, 'b' : 20, 'c' : 30}
print (pd.Series(d))

data = np.random.randn(5)
index = ['a', 'b', 'c', 'd', 'e']
s = pd.Series(data,index)
print(s)

import pandas as pd

# 列表构成的字典
d = {'one' : [1, 2, 3, 4], 'two' : [4, 3, 2, 1]}

df1 = pd.DataFrame(d) # 未指定索引
df2 = pd.DataFrame(d, index=['a', 'b', 'c', 'd']) # 指定索引

print(df1)
print(df2)

import pandas as pd

# 带字典的列表
d = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]

df = pd.DataFrame(d)

print(df)

dd = pd.read_excel('123.xlsx')
print(dd[dd['学号']>=2017103169].head())
#print(dd[:3])

d = [('A', [1, 2, 3]), ('B', [4, 5, 6])]
c = ['one', 'two', 'three']

df = pd.DataFrame.from_items(d,columns=c,orient='index')
print(df)
df.insert(3,'four',[10,20])
print(df)

wp = pd.Panel(np.random.randn(2,5,4),items=['Item1', 'Item2'],
              major_axis=pd.date_range('11/12/2017', periods=5),
              minor_axis=['A', 'B', 'C', 'D'])
print(wp['Item1'])
print(wp.to_frame())'''
df = pd.read_csv('1.csv')
print(df)

df = pd.read_table("1.txt", sep=',') #读取 txt 文件
print(df)

df = pd.read_csv("1.csv") #自定义列索引名称。
print(df)

print(df.head(5))
print(df.tail(5))
print(df.count())
s = pd.Series(np.random.randint(0,9,100))
print(s.value_counts())
print(df.sum())

dd = pd.DataFrame(data={'one':[1,2,3],'two':[4,5,6],'three':[7,8,9]},index=['a','b','c'])
print(dd)
print(dd.reindex(index=['a','b','c'],columns=['one','two','three']))
print(dd.sort_values(by='one',ascending=False))