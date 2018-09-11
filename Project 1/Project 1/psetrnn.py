import numpy as np
h=0
c=0
xlist=[0,0,1,1,1,0]
output=[]
flist=[]
ilist=[]
olist=[]
clist=[]
output=[]
for x in xlist:
    f=sigmoid(-100)
    flist.append(f)
    i=sigmoid(100*x+100)
    ilist.append(i)
    o=sigmoid(100*x)
    olist.append(0)
    c=f*c+i*np.tanh(-100*h+50*x)
    clist.append(c)
    h=o*np.tanh(c)
    output.append(h)

print('h values = ', output, '\n')
print('f values = ', flist, '\n')
print('i values = ', ilist, '\n')
print('o values = ', olist, '\n')
print('c values = ', clist, '\n')

print('DONE')

def sigmoid(x):
    return (1/(1+np.exp(-x)))