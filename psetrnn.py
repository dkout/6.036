import numpy as np

def sigmoid(x):
    return (1/(1+np.exp(-x)))



def rnn(xlist):
    h=0
    c=0
    output=[]
    flist=[]
    ilist=[]
    olist=[]
    clist=[]
    output=[]
    Wch=-100
    Wcx=50

    for x in xlist:
        f=sigmoid(-100)
        flist.append(f)
        i=sigmoid(100*x+100)
        ilist.append(i)
        o=sigmoid(100*x)
        olist.append(o)
        c=f*c+i*np.tanh(-Wch*h+Wcx*x)
        clist.append(c)
        h=round(o*np.tanh(c))
        output.append(h)

    print('h values = ', output, '\n')
    print('f values = ', flist, '\n')
    print('i values = ', ilist, '\n')
    print('o values = ', olist, '\n')
    print('c values = ', clist, '\n')

    print('DONE')

