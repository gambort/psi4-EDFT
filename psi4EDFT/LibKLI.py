import numpy as np
import psi4

np.set_printoptions(precision=5, suppress=True)


class GridHelper:
    def __init__(self, wfn, Ca=None):
        self.wfn = wfn
        
        self.VPot = wfn.V_potential()
        self.PointsFunction = self.VPot.properties()[0]



if __name__=="__main__":
    if False:
        MolStr = """
0 1 

He 0.0 0.0 0.0
He 4.0 0.0 0.0
"""
    else:
        MolStr = """
0 3

O 0.00 0.00 0.00
O 2.40 0.00 0.00
"""

    mol = psi4.geometry(MolStr)
    psi4.set_options({
        'basis': 'def2-tzvp',
        'reference': 'uhf',
#        'dft_block_min_points': 200,
#        'dft_block_max_points': 1000,
#        'dft_basis_tolerance': 1.e-8,
    })
    e, wfn = psi4.energy("pbe", return_wfn=True)

    NBas = wfn.nmo()
    
    NOrb = (wfn.nalpha()+wfn.nbeta())//2 + 1 # Up to LUMO
    NOcc = NOrb-1

    fk = np.zeros(NOrb)
    fk[:NOcc] = 2. # all double occ
    
    C = wfn.Ca().to_array(dense=True)[:,:NOrb]
    D = np.einsum('pi,i,qi->pq', C, fk, C, optimize=True)
    
    VPot = wfn.V_potential()
    PointsFunction = VPot.properties()[0]

    kArr = np.array(range(NOrb))
    
    NBlocks = VPot.nblocks()
    NTot = 0
    NEl = 0.

    WA = [np.zeros((NBas, NBas)) for k in range(NOrb)]
    SA = np.zeros((NBas, NBas))
    for b in range(NBlocks):
        CurrBlock = VPot.get_block(b) # Block of points
        PointsFunction.compute_points(CurrBlock) # Update the points function
        
        NPoints = CurrBlock.npoints() # Number points in block
        NTot +=  NPoints

        LocalBas = np.array(CurrBlock.functions_local_to_global()) # Local basis functions
        NLocal = LocalBas.shape[0] # Number local basis functions
        
        w = np.array(CurrBlock.w()) # Quadrature weights in block on grid
        phiLocal = PointsFunction.basis_values()["PHI"].np[:NPoints, :NLocal] # Basis on grid

        # Test the overlap integrals
        SA[(LocalBas[:,None], LocalBas)] += np.einsum('xp,xq,x->pq', phiLocal, phiLocal, w)
        
        #DLocal = D[(LocalBas[:,None], LocalBas)] # Density matrix in local basis
        #n_r = np.einsum('xp,pq,xq->x', phiLocal, DLocal, phiLocal, optimize=True) # Density on grid
        n_r = 0.
        for k in range(NOrb):
            CkLocal = C[LocalBas,k]
            phik_r = np.dot(phiLocal, CkLocal)
            n_r += fk[k]*phik_r**2
        NEl += np.dot(n_r, w) # Add number of electrons

        for k in range(NOrb):
            break
            CkLocal = C[LocalBas,k]
            phik_r = np.dot(phiLocal, CkLocal)
            nk_r = phik_r**2
            wk_r = nk_r / n_r

            wk_rw = wk_r*w
            T = np.einsum('xp,xq,x->pq', phiLocal, phiLocal, wk_rw, optimize=True)
            WA[k][(LocalBas[:,None], LocalBas)] += T

        print("b = %3d N = %4d, Basis = %5d [%2d%%]"%(b, NPoints, NLocal, NLocal/NBas*100.))

        
    print("Total = %6d, Average = %6.1f, NEl = %10.5f [ %.2f]"%(NTot, NTot/NBlocks, NEl, fk.sum()))
    print(WA[0][:5,:5])
    print(SA[:5,:5])
    print(wfn.S().to_array(dense=True)[:5,:5])
