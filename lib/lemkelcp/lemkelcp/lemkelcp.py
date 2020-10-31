import numpy as np
import pdb

class lemketableau:
    def __init__(self,M,q,maxIter = 100):
        n = len(q)
        self.T = np.hstack((np.eye(n),-M,-np.ones((n,1)),q.reshape((n,1))))
        self.n = n
        self.wPos = list(range(n))
        self.zPos = list(range(n,2*n))
        self.W = 0
        self.Z = 1
        self.Y = 2
        self.Q = 3
        TbInd = np.vstack((self.W*np.ones(n,dtype=int),
                           np.arange(n,dtype=int)))
        TnbInd = np.vstack((self.Z*np.ones(n,dtype=int),
                            np.arange(n,dtype=int)))
        DriveInd = np.array([[self.Y],[0]])
        QInd = np.array([[self.Q],[0]])
        self.Tind = np.hstack((TbInd,TnbInd,DriveInd,QInd))
        self.maxIter = maxIter
        
        
    def lemkeAlgorithm(self):
        initVal = self.initialize()
        if not initVal:
            return np.zeros(self.n),0,'Solution Found'
        
        for k in range(self.maxIter):
            stepVal = self.step()
            if self.Tind[0,-2] == self.Y:
                # Solution Found
                z = self.extractSolution()
                return z,0,'Solution Found'
            elif not stepVal:
                return None,1,'Secondary ray found'
            
        return None,2,'Max Iterations Exceeded'
        
    def initialize(self):
        q = self.T[:,-1]
        minQ = np.min(q)
        if minQ < 0:
            ind = np.argmin(q)
            self.clearDriverColumn(ind)
            self.pivot(ind)
            
            return True
        else:
            return False
            
    def step(self):
        q = self.T[:,-1]
        a = self.T[:,-2]
        ind = np.nan
        minRatio = np.inf
        for i in range(self.n):
            if a[i] > 0:
                newRatio = q[i] / a[i]
                if newRatio < minRatio:
                    ind = i
                    minRatio = newRatio
                    
        if minRatio < np.inf:
            self.clearDriverColumn(ind)
            self.pivot(ind)
            return True
        else:
            return False
        
    def extractSolution(self):
        z = np.zeros(self.n)
        q = self.T[:,-1]
        for i in range(self.n):
            if self.Tind[0,i] == self.Z:
                z[self.Tind[1,i]] = q[i]
        return z
    def partnerPos(self,pos):
        v,ind = self.Tind[:,pos]
        if v == self.W:
            ppos = self.zPos[ind]
        elif v == self.Z:
            ppos = self.wPos[ind]
        else:
            ppos = None
        return ppos
        
    def pivot(self,pos):
        ppos = self.partnerPos(pos)
        if ppos is not None:
            self.swapColumns(pos,ppos)
            self.swapColumns(pos,-2)
            return True
        else:
            self.swapColumns(pos,-2)
            return False
            
        
    
    def swapMatColumns(self,M,i,j):
        Mi = np.array(M[:,i],copy=True)
        Mj = np.array(M[:,j],copy=True)
        M[:,i] = Mj
        M[:,j] = Mi
        return M
    
    def swapPos(self,v,ind,newPos):
        if v == self.W:
            self.wPos[ind] = newPos % (2*self.n+2)
        elif v == self.Z:
            self.zPos[ind] = newPos % (2*self.n+2)
            
    
    def swapColumns(self,i,j):
        iInd = self.Tind[:,i]
        jInd = self.Tind[:,j]
        
        v,ind = iInd
        self.swapPos(v,ind,j)
        v,ind = jInd
        self.swapPos(v,ind,i)
        
        self.Tind = self.swapMatColumns(self.Tind,i,j)
        self.T = self.swapMatColumns(self.T,i,j)
     
    
    def clearDriverColumn(self,ind):
        a = self.T[ind,-2]
        self.T[ind] /= a
        for i in range(self.n):
            if i != ind:
                b = self.T[i,-2]
                self.T[i] -= b * self.T[ind]
                
        
    def ind2str(self,indvec):
        v,pos = indvec
        if v == self.W:
            s = 'w%d' % pos
        elif v == self.Z:
            s = 'z%d' % pos
        elif v == self.Y:
            s = 'y'
        else:
            s = 'q'
        return s
    
    def indexStringArray(self):
        indstr = np.array([self.ind2str(indvec) for indvec in self.Tind.T],dtype=object)
        return indstr
        
    def indexedTableau(self):
        indstr = self.indexStringArray()
        return np.vstack((indstr,self.T))
    def __repr__(self):
        IT = self.indexedTableau()
        return IT.__repr__()
    def __str__(self):
        IT = self.indexedTableau()
        return IT.__str__()
    
def lemkelcp(M,q,maxIter=100):
    """
    sol = lemkelcp(M,q,maxIter)

    Uses Lemke's algorithm to copute a solution to the 
    linear complementarity problem:

    Mz + q >= 0
    z >= 0 
    z'(Mz+q) = 0

    The inputs are given by:

    M - an nxn numpy array
    q - a length n numpy array 
    maxIter - an optional number of pivot iterations. Set to 100 by default

    The solution is a tuple of the form:

    z,exit_code,exit_string = sol

    The entries are summaries in the table below:
    
    |z                | exit_code | exit_string               | 
    -----------------------------------------------------------
    | solution to LCP |    0      | 'Solution Found'          |
    | None            |    1      | 'Secondary ray found'     |
    | None            |    2      | 'Max Iterations Exceeded' |    
    """

    tableau = lemketableau(M,q,maxIter)
    return tableau.lemkeAlgorithm()
