def yh(n):
    L = [1]
    while n>0:
        L = [1]+[x+y for x,y in zip(L,L[1:])]+ [1]
        yield L
        n-=1


for t in yh(5):
    print(t)


L1 = []
L2 = []
for i in range(1,11):
    L1.append(i)
    L2.append(i+1)
print(L1)


L = zip(L1,L2)
print (L)
x = [1, 2, 3]
y = [4, 5, 6, 7]
xy = zip(x, y)
print (xy)
x = [1, 2, 3]

y = [4, 5, 6]

z = [7, 8, 9]

xyz = zip(x, y, z)

print (xyz)

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
