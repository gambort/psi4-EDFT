import psi4
import numpy as np
np.set_printoptions(precision=5, suppress=True)

def GetVStep(wfn, C = None, fw = [2.]):
    fw = np.array(fw)
    Nfw = fw.shape[0]

    kMax = 0
    for k in range(Nfw):
        if fw[k]>1e-5: kMax=k

    ffw = fw*1.
    ffw[kMax] = 0.
        
    Vpot = wfn.V_potential()

    # Grab a "points function" to compute the Phi matrices
    points_func = Vpot.properties()[0]

    # Grab a block and obtain its local mapping
    block = Vpot.get_block(1)
    npoints = block.npoints()
    lpos = np.array(block.functions_local_to_global())

    # Copmute phi, note the number of points and function per phi changes.
    phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

    if C is None:
        C = wfn.Ca().to_array(dense=True)

    Df  = np.einsum('pi,qi,i->pq', C[:,:Nfw],C[:,:Nfw],fw)
    Dff = np.einsum('pi,qi,i->pq', C[:,:Nfw],C[:,:Nfw],ffw)
    

    V = np.zeros_like(Df)
    N_e = 0.0
    Nf_e = 0.0

    rho = []
    points_func = Vpot.properties()[0]
    superfunc = Vpot.functional()

    # Loop over the blocks
    for b in range(Vpot.nblocks()):

        # Obtain block information
        block = Vpot.get_block(b)
        points_func.compute_points(block)
        npoints = block.npoints()
        lpos = np.array(block.functions_local_to_global())


        # Obtain the grid weight
        w = np.array(block.w())

        # Compute phi!
        phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

        # Build a local slice of D
        lDf  = Df [(lpos[:, None], lpos)]
        lDff = Dff[(lpos[:, None], lpos)]

        # Copmute rho and rhof
        rhof  = np.einsum('pm,mn,pn->p', phi, lDf, phi)
        rhoff = np.einsum('pm,mn,pn->p', phi, lDff, phi)
        
        # Computer electron number
        N_e += np.dot(w, rhof)
        Nf_e += np.dot(w, rhoff)

        # Compute the potential
        v_rho_a = rhoff/rhof
        Vtmp = np.einsum('pb,p,p,pa->ab', phi, v_rho_a, w, phi)

        # Add the temporary back to the larger array by indexing, ensure it is symmetric
        V[(lpos[:, None], lpos)] += 0.5 * (Vtmp + Vtmp.T)


    #print("Electron number %16.8f  %16.8f"%(N_e, Nf_e))
    
    return V


if __name__ == "__main__":
    psi4.geometry("""
0 1

He
symmetry c1""")
    psi4.set_options({
        'basis':'def2-tzvp',
    })
    E, wfn = psi4.energy("pbe0", return_wfn=True)
    
    fw = np.array([1.8,0.2])
    V = GetVStep(wfn, fw=fw)
    S = wfn.S().to_array(dense=True)

    
    print(V[:7,:7])
    print(S[:7,:7])
