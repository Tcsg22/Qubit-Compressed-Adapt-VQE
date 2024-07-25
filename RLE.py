import numpy as np
from bitarray import bitarray
import random

class generator:
    '''Generator class

    Input:
    ------
    M: the number of modes

    N: the number of particles

    Q: the guessed compression

    maxiter: the maximal number of trials

    Ouput:
    ------
    multiple functions that generates the generator matrix
    '''


    def __init__(self, M, N, Q, maxiter):
        self.M = M
        self.N = N
        self.Q = Q
        self.hd =N*2
        self.lc = int(np.ceil(Q/2))


        s=[]
        for i in self.bitmasks(M, N):
            s.append(i.to01())

        lc=int(np.ceil(Q/2))
        if lc%2==1:
               lc-=1

        self.s = s
        self.lc = lc


        self.maxiter = maxiter

        #self.items = 0

    'recursive'
    def bitmasks(self, M, N):
        if N < M:
            if N > 0:
                for x in self.bitmasks(M-1,N-1):
                    yield bitarray([1]) + x
                for x in self.bitmasks(M-1,N):
                    yield bitarray([0]) + x
            else:
                yield M * bitarray('0')
        else:
            yield M * bitarray('1')


    def modtest2(self):
        '''generator matrix with substantially simpler check
        '''
    #    s10=np.array(to10(s))
        aa=0
        for r in range(self.maxiter):
            # sn=np.mod(s10,r)
            if r%10==0:
                print(r,aa)
            B,a=self.Bg()
            ps=[]
            for i in self.s:
                v=self.to10(list(i))
                v=B.dot(v)%2
                vs=''
                for j in v:
                    if j==0:
                        vs+='0'
                    else:
                        vs+='1'
                ps.append(vs)
            aa+=len(self.s)

            g = len(set(ps))
            f = len(self.s)
            print('Injectivity: ', g, f, g/f)
            if g==f:
                return B#ps,B,a,aa


    def modtest_small(self):
        c=0
        for r in range(self.maxiter):
            if r%20==0:
                print(r,c)
            B,a=self.Bg()
            ps=[]
            for i in self.s:
                v=self.to10(list(i))
                v=B.dot(v)%2
                vs=''
                for j in v:
                    if j==0:
                        vs+='0'
                    else:
                        vs+='1'
                ps.append(vs)
            sn=to10(ps)
            if len(set(sn))>c and max(sn)<2**self.Q:
                print(r,len(self.s),len(set(sn)),(len(set(sn)))/len(self.s))
                c=len(set(sn))
                sf=sn.copy()
                Bf=B.copy()
                if len(self.s)==c:
                    break
                
        return Bf, #sf,Bf,a

    def Bg(self):
        # while 1:
        B=list(np.eye(self.Q,dtype=np.int32))
        a=[]
        # for j in range(int(Q/N)-1):
        #     tmp=np.zeros(Q,dtype=np.int32)
        #     tmp[j*N:j*N+2*N]=1
        #     for i in range(len(tmp)):
        #         atmp="".join(map(str, tmp))
        #     B.append(tmp)
        #     a.append(atmp)
                
            
        while len(B)<self.M:
            tmp=np.zeros(self.Q,dtype=np.int32)
            ls=random.sample(list(range(int(self.Q))),self.lc)
            tmp[ls]=1
            for i in range(len(tmp)):
                atmp="".join(map(str, tmp))
            c=[self.hd]
            for i in range(len(a)):
                c.append(sum(np.array(list(a[i]))!=np.array(list(atmp))))
            while min(c)<self.hd:
                tmp=np.zeros(self.Q,dtype=np.int32)
                ls=random.sample(list(range(int(self.Q))),self.lc)
                tmp[ls]=1
                for i in range(len(tmp)):
                    atmp="".join(map(str, tmp))
                c=[self.hd]
                for i in range(len(a)):
                    c.append(sum(np.array(list(a[i]))!=np.array(list(atmp))))
            B.append(tmp)
            a.append(atmp)
            
        B=np.array(B).T
        return B,a

    def to10(self, s):
        a=[]
        for i in s:
            a.append(int(i,2))
        return np.array(a)

def encoder(orb, e, Q):
    '''Compute the compression matrix with the molecule information.

    Input:
    ------
    mol: molecule object.

    Q: the guessed qubit number.

    freeze: freezing core orbitals.

    RHF: restricted Hartree-Fock.

    Output:
    -------
    A: a (Q, M) matrix
    '''
    
    g = generator(orb, e, Q, 5000)
    A = g.modtest2()
    
    return A
