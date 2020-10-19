import numpy as np
import scipy.linalg as la

class DegenHelper:
    def __init__(self, C=None, H=None, epsilon=1e-5):
        if not(C is None):
            self.NTot = C.shape[1]
            kIndx = []
            for k in range(self.NTot):
                kIndx += [np.argwhere(np.abs(C[:,k])>epsilon).reshape((-1,))]
        else:
            F = H
            if F is None:
                quit()
            
            dF = np.abs(np.diag(F))
            x = np.abs(F)/np.sqrt(np.outer(dF,dF))

            self.NTot = x.shape[1]

            kIndx = []
            for k in range(self.NTot):
                kIndx += [np.argwhere(x[:,k]>epsilon).reshape((-1,))]

        ######################################
        # Block things
        self.kAll = {}
        for I in kIndx:
            k0 = I.min()
            if not(k0 in self.kAll):
                self.kAll[k0] = I
            elif len(I)>len(self.kAll[k0]):
                for k in I:
                    if not(k in self.kAll[k0]):
                        self.kAll[k0] = np.hstack((self.kAll[k0],k))
        for k0 in self.kAll:
            self.kAll[k0] = np.sort(self.kAll[k0])
        ######################################
        # Remove duplicates
        Master = np.zeros((self.NTot,),dtype=int)
        for k0 in sorted(list(self.kAll)):
            I = self.kAll[k0]
            kUnique = []
            for i in I:
                if Master[i]==0:
                    kUnique += [i]
                    Master[i]=1

            if len(kUnique)>0:
                self.kAll[k0] = np.array(kUnique, dtype=int)
            else: self.kAll.pop(k0)
        # Test
        kCheck = np.zeros((self.NTot,), dtype=int)
        for k in self.kAll:
            kCheck[self.kAll[k]]=1

        NMissing = np.sum(kCheck) - self.NTot
        if not(NMissing==0):
            print("Number missing elements = %d"%(NMissing))
            quit()

        if True and len(self.kAll)>1:
            print("Symmetry mappings:")
            for k0 in self.kAll:
                print("%3d :"%(k0) + \
                      ", ".join(["%3d"%(x) for x in self.kAll[k0]]))

        self.EigIndx = None
        
    def Solve(self, F, Sort="Pre"):
        # Solves the eigenvalue equaiton with the degeneracies
        if not(F.shape==(self.NTot,self.NTot)):
            print("Does not have the right shape")
            return None

        w = np.zeros(F.shape[0])
        v = np.zeros(F.shape)


        for k0 in self.kAll:
            kp = self.kAll[k0]

            Fp = F[kp[:,None],kp]
            wp, vp = la.eigh(Fp)

            w[kp] = wp
            v[kp[:,None],kp] = vp

        if Sort is None or Sort[0] in ("N", "n"):
            return w, v # Not recommended

        if Sort[0] in ("A", "a") or self.EigIndx is None:
            ii = np.argsort(w)
            self.EigIndx = ii

        w = w[self.EigIndx]
        v = v[:,self.EigIndx]
            
        return w,v
            

if __name__=="__main__":
    F1 = np.ones((2,2)) + np.eye(2)*10.
    F2 = np.ones((3,3)) + np.eye(3)*5.
    F3 = np.ones((2,2))*3.
    F = la.block_diag(F1,F2,F3)


    
    
    DH = DegenHelper(F)

    w,v=DH.Solve(F)
    print(w)
    print(v)
    print(np.dot(v,np.diag(w)).dot(v.T))
    
