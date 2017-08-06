import os
import math
import types
from collections import Iterator
from  types import MethodType
for i in os.listdir('D:'):
    print(i)

L = ['What',18,'YEEE','Hu',None,True,'MAX','LI','Hello']
t = [x.lower() if isinstance(x,str) else x for x in L]

print(t)

def fib(n):
    a,b,i = 0,1,0
    while i<n:
        a,b = b,a+b
        i += 1
    return b
print(fib(2))

def odd():
    print('step 1')
    yield 1
    print('step 2')
    yield 3
    print('step 3')
    yield 5

o = odd()
next(o)
next(o)
next(o)

def g(n):
    list = [1]
    while n > 0:
        yield list
        list = [1] + [x + y for x, y in zip(list, list[1:])] + [1]
        n -= 1
    return

for t in g(5):
    print(t)


def my_gen():
    n = 1
    print('This is printed first')
    # Generator function contains yield statements
    yield n

    n += 1
    print('This is printed second')
    yield n

    n += 1
    print('This is printed at last')
    yield n

for item in my_gen():
    print(item)
print("-----------------------------")
def yh(n):
    L = [1]
    m = 1
    yield L
    while m<=n:
        m += 1
        g = []
        if m<=2:
            g = [1,1]
        else:
            l = len(L)-1
            for i in range(0,l):
                g.append((L[i]+L[i+1]))
            g = [1] + g + [1]
        L = g
        yield L
for t in yh(10):
    print(t)

L = [x*x for x in range(1,11)]
print(isinstance((x for x in range(10)), Iterator))



from math import sqrt
def cc(a,b,*p):
    n = a+b
    for i in p:
        n=i(n)
    return n
print (cc(15,1,sqrt,sqrt))

from math import sqrt

def do_sth(x=[],*func):
    return [f(x_k) for x_k in x for f in func]
def ff(n):
    return -n
print(do_sth([1,2,4,9],ff,sqrt))

L = map(ff,[x for x in range(1,11)])
print(list(L))
print("--------")

'''
def mm(l):
    ans = 0
    for i in l:
        if i>ans: i = -i
    return l

print (list(map(mm,([x for x in range(1,10)]))))

'''


LL = []
for i in range(1,11):
    LL.append(i)
LL = list(map(str,LL))
print(LL)

def xx(l):
    ans = 0
    for i in l:
        ans+=i
    return ans

from functools import reduce
def xx(a,b): return a+b
LL = [x for x in range(1,10)]
print(reduce(xx,LL))

def xx(a,b):
    return a*10+b

print(reduce(xx,LL))

def sr(str):
    str = str[:1].upper()+str[1:].lower()
    return str
print(sr('SfTc'))


L1 = ['adam', 'LISA', 'barT']
def normalize(str):
    return str.capitalize()
L2 = list(map(normalize, L1))
print(L2)

'''
capitalize() 首字母大写，其余全部小写 

upper() 全转换成大写

lower() 全转换成小写

title()  标题首字大写，如"i love python".title()  "I love python"
'''


def prod(l):
    def aa(a, b):
        return a * b
    return reduce(aa,l)
print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))

def str2float(s):
    def fn(x, y):
        return x * 10 + y
    def fa(x, y):
        return x * 0.1 + y
    def char2num(s):
        return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]
    dot = s.find('.')
    return reduce(fn, map(char2num, s[:dot]))+0.1*reduce(fa, map(char2num, s[:dot:-1]))

print('str2float(\'123.456\') =', str2float('123.456'))
s ='123.456'
print(s[:3:-1])

def str2float(s):
    def char2num(s):
        return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]
    def fn(x, y):
        return x * 10 + y
    dot = s.find('.')
    strlen = len(s)-dot-1
    ss = s[:dot]+s[dot+1:]
    return reduce(fn,map(char2num,ss))/(10 ** strlen)
print('str2float(\'123.456\') =', str2float('123.456'))
print('str2float(\'12345.6\') =', str2float('12345.6'))

def odd_iter():
    n = 1
    while True:
        n += 2
        yield n

def yu(n):
    return lambda x:x%n > 0

def prime():
    yield 2
    it = odd_iter()
    while True:
        n = next(it)
        yield n
        it = filter(yu(n),it)
for n in prime():
    print(n)
    if n>100 : break

from functools import reduce
l = [1,2,3,5,-9,0,45,-99]
print(list(map(abs,l)))

l = [1,2,3,5,-9,0,45,-99]
print(list(filter(lambda x:x<0,l)))

from functools import reduce
l = [1,2,3,5,-9,0,45,-99]
print(reduce(lambda x,y:x+y,l))

for i in range(1,10):
        print(' '.join(["%d * %d = %2d "%(j,i,i*j) for j in range(1,i+1)]))

def ad(n):
    return lambda x:x+n
aa = ad(3)
print(aa(5))

def ini(s):
    return str(s)==str(s)[::-1]
ll = list(filter(ini,range(1,1001)))
print(ll)

L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]

print (sorted(L, key=lambda x:x[0]))

L = [36, 5, -12, 9, -21]
print(sorted(L,key = abs))
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
print(sorted(L,key=lambda x :x[0]))
print(sorted(L,key=lambda x :x[1],reverse=True))

ll = list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
print(ll)

seq = []
for i in range(5):
    seq.append(i*i)
print (seq)

def xx(a):
    return lambda b:a+b

ll = [x for x in range(1,5)]
print(reduce(lambda x,y:x+y,ll))
import functools
int2 = functools.partial(int,base=2)
print(int2('10000000'))
def cal(*num):
    ans = 0
    for i in num:
        ans += i
    return ans
ll = [i for i in range(1,11)]
print(cal(*ll))
def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)
tu = {'gender': 'M', 'job': 'Engineer'}
person('mayuchi',24,**tu)

def f1(a, b, c=0, *args, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)
def f2(a, b, c=0, *, d, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'd =', d, 'kw =', kw)
args = (1, 2, 3)
kw = {'d': 88, 'x': '#'}
f1(*args, **kw)
f2(*args, **kw)

class fruit(object):
    def __init__(self,name,price,weight):
        self.name = name
        self.price = price
        self.weight = weight
    def fruit_print(self):
        print("%s : %s %s"%(self.name,self.price,self.weight))

apple = fruit("apple",13,200)
apple.fruit_print()
print(apple.name)

class Student(object):
    def __init__(self,name,score):
        self.__name = name
        self.__score = score
    def print_score(self):
        print("%s: %s"%(self.__name,self.__score))
    def getname(self):
        return self.__name
    def getscore(self):
        return self.__score
    def setscore(self,score):
        if 0<= score <=100 :
            self.__score = score
        else:
            print("score Error!!!!")

mayc = Student('mayc199',99)
print(mayc.getname(),mayc.getscore())
mayc.setscore(88)
print(mayc.getscore())

class animal(object):
    def run(self):
        print('Animal is running...')
class dog(animal):
    def run(self):
        print('dog is running...')
class cat(animal):
    def run(self):
        print('cat is runnung...')


ca = cat()
ca.run()
print(isinstance(ca,animal))

def run_twice(an):
    an.run()
    an.run()

class tor(animal):
    def run(self):
        print('tor is running slowly..')
run_twice(tor())

class Ptimer(object):
    def run(self):
        print('Start..')

run_twice(Ptimer())

print(type('123112233'))

def fn():
    pass
print(isinstance(type(fn),types.FunctionType))
print(types.FunctionType==type(fn))
print(type(abs)==types.BuiltinFunctionType)

class animal(object):
    pass

class Dog(animal):
    pass

class Husky(Dog):
    pass

a = animal()
d = Dog()
h = Husky()

print(isinstance(h,animal))

class studentt(object):
    pass
    #__slots__ = ('name','age')
s = studentt()
s.name = 'mayc199'
print(s.name)


def set_age(self, age):
    self.age = age

studentt.set_age = set_age
s.set_age(100)
print(s.age)

class sblx(object):
    __slots__ = ('name','na')

s = sblx()
s.name = 'sb'
s.na = 'jb'

print(s.name,s.na)

class genas(sblx):
    pass

ss = genas()
ss.score = 0
print(ss.score)

class person(object):
    __slots__ = ('name', 'age')


class stu(person):
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def printt(self):
        print('%s\'s age is %s'%(self.name,self.age))


s = stu('hahah','12')
s.printt()

'''
ss = person()
ss.score = 60
print(person.score)
'''
class stud(object):
    @property
    def score(self):
        return (self._score)
    @score.setter
    def score(self,value):
        if not isinstance(value,int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value




s = stud()
s.score = 79
print(s.score)

class Screen(object):
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self,va):
        if not isinstance(va, int) or va<0:
            raise ValueError('Error!')
        else:
            self._width = va

    @property
    def height(self):
        return self._height
    @height.setter
    def height(self,va):
        if not isinstance(va,int) or va<0:
            raise ValueError('Error!')
        else :
            self._height = va

    @property
    def resolution(self):
        return self._height*self._width


s = Screen()
s.width = 1024
s.height = 768
print(s.resolution)
assert s.resolution == 786432, '1024 * 768 = %d ?' % s.resolution