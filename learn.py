import os
import math
import types
from collections import Iterator
from  types import MethodType
from enum import Enum
from io import StringIO
from io import BytesIO
import os


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
'''
from tkinter import *

def cb1():
    print('button1 clicked')
def printEventInfo(event):
    print('event.time = ', event.time)
    print('event.type = ', event.type)
    print('event.WidgetId = ', event.widget)
    print('event.KeySymbol = ', event.keysym)
def cb3():
    print('button3 clicked')
root = Tk()
b1 = Button(root, text='Button1', command=cb1)
b3 = Button(root, text='Button3', command=cb3)
b1.pack()

b3.pack()

root.mainloop()
'''
class Animal(object):
    pass

# 大类:
class Mammal(Animal):
    pass

class Runnable(object):
    def run(self):
        print('Running...')

class Flyable(object):
    def fly(self):
        print('Flying...')

class Bird(Animal):
    pass

# 各种动物:
class Dog(Mammal,Runnable):
    pass

class Bat(Mammal,Flyable):
    pass

class Parrot(Bird):
    pass

class Ostrich(Bird):
    pass

class Student(object):
    @property
    def nn(self):
        return self.name
    @nn.setter
    def nn(self,value):
        self.name = value

s = Student()
s.name = '2333'
print(s.name)

class Student(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return ( 'Student object (name: %s)' % self.name)
    __repr__ = __str__

s = Student('mike')

class fib(object):
    def __init__(self):
        self.a,self.b = 0,1
    def __iter__(self):
        return self
    def __next__(self):
        self.a,self.b = self.b,self.a+self.b
        if self.b>10000:
            raise StopIteration()
        return  self.a

class Fib(object):
    def __getitem__(self, n):
        if isinstance(n,int):
            a,b = 1,1
            for x in range(n):
                a,b = b,a+b
            return a
        elif isinstance(n,slice):
            t = n.start
            p = n.stop
            if t is None:
                t = 0
            a,b = 1,1
            L = []
            for x in range(p):
                if x >= t:
                    L.append(a)
                a,b = b,a+b
            return L

f = Fib()
print("----------------")
print(f[9])
print(f[:10])

class Chain(object):

    def __init__(self, path=''):
        self._path = path

    def __getattr__(self, path):
        return Chain('%s/%s' % (self._path, path))

    def __str__(self):
        return self._path

    __repr__ = __str__
ts = '23'
print(Chain().status.user.timeline.list)

class person(object):
    def __init__(self,name,gender):
        self.name = name
        self.gender = gender
    def __str__(self):
        return ('%s: %s'%(self.name,self.gender))

class student(person):
    def __init__(self,name,gender,score):
        super(student,self).__init__(name,gender)
        self.score = score
    def __str__(self):
        return '%s: %s %s'%(self.name,self.gender,self.score)
ss = student('mike','male',88)
print(ss)

class ss(object):
    def __init__(self):
        self.name = 'mike'
    def __getattr__(self, item):
        if item == 'score':
            return 99
s = ss()
print(s.name)
print(s.score)

class ent(object):
    def __init__(self,name):
        self.name = name
    def __call__(self):
        print('my name is %s' % self.name)

ss = ent('mek')
ss()
class Student(object):
    def __init__(self):
        self.name = 'Michael'
    def __getattr__(self, attr):
        if attr=='score':
            return 99
        if attr=='age':
            return lambda: 25
        raise AttributeError('\'Student\' object has no attribute \'%s\'' % attr)

s = Student()
print(s.name)
print(s.score)
print(s.age())
# AttributeError: 'Student' object has no attribute 'grade'
#print(s.grade)

from enum import Enum

class Color(Enum):
    red = 1
    orange = 2
    yellow = 3
    green = 4
    blue = 5
    indigo = 6
    purple = 7
print(Color['red'])
for color in Color:
    print(color)
for color in Color.__members__.items():
    print(color)


def fn(self,name = 'world'):
    print('hello,%s'%name)

Hello = type('Hello',(object,),dict(hello = fn))
h = Hello()
h.hello()
'''
class ModelMetaclass(type):

    def __new__(cls, name, bases, attrs):
        if name=='Model':
            return type.__new__(cls, name, bases, attrs)
        print('Found model: %s' % name)
        mappings = dict()
        for k, v in attrs.items():
            if isinstance(v, Field):
                print('Found mapping: %s ==> %s' % (k, v))
                mappings[k] = v
        for k in mappings.keys():
            attrs.pop(k)
        attrs['__mappings__'] = mappings # 保存属性和列的映射关系
        attrs['__table__'] = name # 假设表名和类名一致
        return type.__new__(cls, name, bases, attrs)
class ListMetaclass(type):
    def __new__(cls, name, bases, attrs):
        attrs['add'] = lambda self, value: self.append(value)
        return type.__new__(cls, name, bases, attrs)
class MyList(list, metaclass=ListMetaclass):
    pass

L = MyList()
L.add(1)
print(L)
class Model(dict, metaclass=ModelMetaclass):
    def __init__(self, **kw):
        super(Model, self).__init__(**kw)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Model' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value

    def save(self):
        fields = []
        params = []
        args = []
        for k, v in self.__mappings__.items():
            fields.append(v.name)
            params.append('?')
            args.append(getattr(self, k, None))
        sql = 'insert into(%s) (%s) values (%s)' % (self.__table__, ','.join(fields), ','.join(params))
        print('SQL: %s' % sql)
        print('ARGS: %s' % str(args))
class User(Model):
    id = IntegerField('id')
    name = StringField('username')
    email = StringField('email')
    passward = StringField('password')


class Field(object):
    def __init__(self, name, column_type):
        self.name = name
        self.column_type = column_type
    def __str__(self):
        return '<%s:%s>' % (self.__class__.__name__,self.name)
class StringField(Field):
    def __init__(self,name):
        super(StringField,self).__init__(name,'varchar(100)')
class IntegerField(Field):
    def __init__(self,name):
        super(IntegerField, self).__init__(name,'bigint')

u = User(id=1234, name='mike', email='23@233.com', passward='password')
u.save()

try:
    print('try...')
    r = 10 / 2
    print('result:', r)
except ZeroDivisionError as e:
    print('except:', e)
finally:
    print('finally...')
print('END')

try:
    print('try...')
    r = 10/0
    print('result: %d'%(r))
except ValueError as e:
    print('ec 1')
    print('ValueError:',e)
except ZeroDivisionError as e:
    print('ec 1')
    print('ZeroDivisionError:',e)
else:
    print('no error!')
finally:
    print('finally..')

def foo(s):
    n = int(s)
    assert n != 0, 'n is zero!'
    return 10 / n

def main():
    foo('0')

#main()


程序中如果到处充斥着assert，和print()相比也好不到哪去。不过，启动Python解释器时可以用-O参数来关闭assert：

$ python3 -O err.py
Traceback (most recent call last):
  ...
ZeroDivisionError: division by zero

import logging
logging.basicConfig(level=logging.INFO)
s = '0'
n = int(s)
logging.info('n = %d' % n)
print(10 / n)
'''
ff = open('123.txt')
print(ff.read())
ff.close()
print('-----------')
try:
    f = open('123.txt','r')
    print(f.read())
except IOError as e:
    print('IOError!',e)
finally:
    if f:
        f.close()
print('----------2')
#with open('123.txt') as f:
#    print(f.read())

#f.close()
f = open('123.txt')
for line in f.readlines():
    print(line.strip()) # 把末尾的'\n'删掉
f.close()
print('-----------1')
f = open('123.txt')
print(f.readlines())
f.close()
print('-----------12')
f = open('123.txt')
LL=[]
for i in f.readlines():
    LL.append(i.strip())
f.close()
print(LL)
'''
with open('123.txt','a') as f:
    f.write('\nhello word!')
print('--------------')
'''
with open('123.txt',) as f:
    with open('123-1.txt','w+') as s:
        for i in f.readlines():
            s.write(i.replace('hello','hi'))
f = StringIO()
f.write('1')
f.write('2')
f.write('3')
f.write('4')
f.write('5')
print(f.getvalue())
ff = StringIO('Hello!\nHi!\nGoodbye!')
while True:
    s = ff.readline()
    if s=='': break
    print(s.strip())

f = BytesIO()
f.write('中文！！'.encode('utf-8'))
print(f.getvalue())

print(os.name)
print(os.environ)
print(os.environ.get('path'))
print(os.path.abspath(''))
os.path.join('D:\Py', 'testdir')
#os.mkdir('D:\Py\\testdir')
#os.rmdir('D:\Py\\testdir')
#os.rename('123-1.txt','123-11.txt')
#os.remove('123-11.txt')
ll = [x for x in os.listdir() if os.path.isfile(x) and os.path.splitext(x)[1]=='.py']
print(ll)
print('------------')
from multiprocessing import Pool
import os, time, random
'''
def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
'''
from datetime import datetime
now = datetime.now()
print(now)
tt = 123456789.0
print(datetime.fromtimestamp(tt))
print(datetime.utcfromtimestamp(tt))

cd = datetime.strptime('2017.8.15 17:04','%Y.%m.%d %H:%M')
print(cd)

now = datetime.now()
cstr = now.strftime('%Y %d of %m %H:%M:%S %a')
print(cstr)

from collections import namedtuple,deque,defaultdict,OrderedDict,Counter
point = namedtuple('point',['x','y'])
p = point(1,2)
print(p.x)
print(p.y)

q = deque(['a', 'b', 'c'])
q.append('x')
print(q)
q.appendleft('y')
print(q)
q.pop()
print(q)
q.popleft()
print(q)

dd = defaultdict(lambda :'N/A')
dd['key'] = 'abc'
print(dd['key'])
print(dd['kk'])
od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
print(od)
od = OrderedDict()
od['z'] = 1
od['y'] = 2
od['x'] = 3
print(od)
print(list(od.keys()))
print(list(od.values()))

class LastUpdatedOrderedDict(OrderedDict):

    def __init__(self, capacity):
        super(LastUpdatedOrderedDict, self).__init__()
        self._capacity = capacity

    def __setitem__(self, key, value):
        containsKey = 1 if key in self else 0
        if len(self) - containsKey >= self._capacity:
            last = self.popitem(last=False)
            print('remove:', last)
        if containsKey:
            del self[key]
            print('set:', (key, value))
        else:
            print('add:', (key, value))
        OrderedDict.__setitem__(self, key, value)

c = Counter()
for ch in 'uhguihbkjbgjkhg':
    c[ch] += 1
print(c)
import base64
print(base64.b64encode(b'binarystring'))
def safe_base64_decode(s):
    return base64.b64decode(s+b'='*(4-(len(s)%4)))
assert b'abcd' == safe_base64_decode(b'YWJjZA=='), safe_base64_decode('YWJjZA==')
assert b'abcd' == safe_base64_decode(b'YWJjZA'), safe_base64_decode('YWJjZA')
print('Pass')

import hashlib
md5 = hashlib.md5()
md5.update('how to use md5 in python hashlib?'.encode('utf-8'))
print(md5.hexdigest())

sh1 = hashlib.sha1()
sh1.update('how to use md5 in python hashlib?'.encode('utf-8'))
print(sh1.hexdigest())


def calc_md5(password):
    cc = str(password)
    md5 = hashlib.md5()
    md5.update(cc.encode('utf-8'))
    return md5.hexdigest()
print(calc_md5(888888))

db = {
    'michael': 'e10adc3949ba59abbe56e057f20f883e',
    'bob': '878ef96e86145580c38c87f0410ad153',
    'alice': '99b1c2188db85afee403b1536010c2c9'
}
def login(user, password):
    u = str(user)
    p = str(password)
    if db[u] ==calc_md5(p):
        return True
    return False
print(login('bob', 'abc999'))

db = {}
def get_md5(password):
    md5 = hashlib.md5()
    md5.update(password.encode('utf-8'))
    return md5.hexdigest()

def register(username, password):
    db[username] = get_md5(password + username + 'the-Salt')
    print('注册成功，请登录')

def login(username, password):
    if db[username] == get_md5(password + username + 'the-Salt'):
        print('登陆成功！')
    elif db[username] == 0:
        print('用户不存在！')
    else:
        print('用户名或密码错误！')

print('请注册')
print('请输入用户名和密码！')
username = input('username = ')
password = input('password = ')
time.sleep(1)
register(username,password)
time.sleep(1)
print('请登入！')
username = input('username = ')
password = input('password = ')
print('正在登录。。。')
time.sleep(1)
login(username,password)


