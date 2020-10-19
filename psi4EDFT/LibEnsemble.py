#!/home/timgould/psi4conda/bin/python
import psi4

import psi4EDFT.TimeKeeper as TK

from psi4EDFT.LibSSOEP import *
from psi4EDFT.LibDegen import *

import os

# Ensure the right number of threads set using MKL_NUM_THREADS
if 'MKL_NUM_THREADS' in os.environ:
    NThreads = int(os.environ['MKL_NUM_THREADS'])
else: NThreads = 4

psi4.core.set_num_threads(NThreads)
try:
    import mkl
    mkl.set_num_threads(NThreads)
except:
    pass

# Then import numpy and scipy
import numpy as np
import scipy.linalg as la


# See:
# http://forum.psicode.org/t/basic-psi4numpy-question-for-the-dft-potential/661/2
# http://www.psicode.org/psi4manual/master/api/psi4.core.VBase.html
# http://www.psicode.org/psi4manual/master/api/psi4.core.LibXCFunctional.html
# http://forum.psicode.org/t/obtain-the-energy-from-dft-at-arbitrary-density-matrix/1884/4

# Helps deal with electron repulsion integrals (ERIs)
# Can do density fitting - but not on rs part
class ERIHelper:
    def __init__(self, wfn, JKFit=True, alpha=None, omega=None):
        self.wfn = wfn
        self.basis = wfn.basisset()
        self.mints = psi4.core.MintsHelper(self.basis)
        self.alpha = alpha
        self.omega = omega
        self.eri = self.mints.ao_eri()
        if not(self.omega is None):
            self.eri_w = self.mints.ao_erf_eri(self.omega)

        self.JKFit = JKFit
        if self.JKFit:
            # Calculate auxilliary basis for JKFit
            self.aux_basis = \
                             psi4.core.BasisSet.build(self.wfn.molecule(),
                                                      "DF_BASIS_SCF", "",
                                                      "JKFIT", self.basis.name())
            self.zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
        
            # Get the density fit business
            # Density fit stuff
            self.SAB = np.squeeze(
                self.mints.ao_eri(self.zero_basis, self.aux_basis,
                                  self.basis, self.basis))
            metric = \
                     self.mints.ao_eri(self.zero_basis, self.aux_basis,
                                       self.zero_basis, self.aux_basis)
            metric.power(-0.5, 1e-14)
            metric = np.squeeze(metric)
            
            # ERI in auxilliary - for speed up
            self.ERIA = np.tensordot(metric, self.SAB, axes=[(1,),(0,)])
            
            if not(self.omega is None):
                # Get the density fit business
                # Need to work out how to do density fit on rs part
                if False:
                    self.SAB_w = np.squeeze(
                        self.mints.ao_erf_eri(self.omega,
                                              self.zero_basis, self.aux_basis,
                                              self.basis, self.basis))
                    metric_w = \
                             self.mints.ao_erf_eri(self.omega,
                                                   self.zero_basis, self.aux_basis,
                                                   self.zero_basis, self.aux_basis)
                    metric_w.power(-0.5, 1e-14)
                    metric_w = np.squeeze(metric_w)

                    # ERI in auxilliary - for speed up
                    self.ERIA_w = np.tensordot(metric_w, self.SAB_w, axes=[(1,),(0,)])
            
        
    def JEH(self, D):
        if not(self.JKFit):
            J = np.tensordot(D, self.eri.np, 2)
        else:
            T = np.tensordot(self.ERIA, D, ((1,2),(0,1)))
            J = np.tensordot(self.ERIA, T, ((0,),(0,)))

        EH = 0.5*np.tensordot(J, D, 2)
            
        return J, EH

    def KEx(self, D, alfactor=0.5, factor=None):
        if factor is None:
            if self.alpha is None:
                return 0.,0.
            factor = -alfactor*self.alpha
            omega = self.omega
        else: omega=None
        
        if omega is None:
            # Need to JKFit this
            if not(self.JKFit):
                K = factor*np.tensordot(D, self.eri.np, ((0,1),(0,2)))
            else:
                T = np.tensordot(self.ERIA, D, ((1,),(0,)))
                K = factor*np.tensordot(self.ERIA, T, ((0,1),(0,2)))
        else:
            AK = self.alpha
            BK = (1. - self.alpha)
            if not(self.JKFit):
                K  = -alfactor*AK*np.tensordot(D, self.eri.np, ((0,1),(0,2)))
                K += -alfactor*BK*np.tensordot(D, self.eri_w.np, ((0,1),(0,2)))
            else:
                T  = np.tensordot(self.ERIA, D, ((1,),(0,)))
                K  = -alfactor*AK*np.tensordot(self.ERIA, T, ((0,1),(0,2)))
                K += -alfactor*BK*np.tensordot(D, self.eri_w.np, ((0,1),(0,2)))
                
        Ex = 0.5*np.tensordot(K, D, 2)

        return K, Ex

    def E_IJKL(self, CI,CJ,CK,CL):
        if not(self.JKFit):
            T = np.dot(self.eri.np, CL)
            T = np.dot(T, CK)
            T = np.dot(T, CJ)
            T = np.dot(T, CI)
            return 0.5*T
        else:
            T1 = np.dot(self.ERIA, CL)
            T1 = np.dot(T1, CK)
            T2 = np.dot(self.ERIA, CJ)
            T2 = np.dot(T2, CI)

            return 0.5*np.dot(T1, T2)

    def Ex_IJKL(self, CI,CJ,CK,CL):
        if self.alpha is None:
            return 0.
        factor = self.alpha
        omega = self.omega
        
        if omega is None:
            return factor*self.E_IJKL(CI,CJ,CK,CL)
        else:
            AK = self.alpha
            BK = (1. - self.alpha)

            Ex  = AK*self.E_IJKL(CI,CJ,CK,CL)
            T = np.dot(CI, self.eri_w.np)
            T = np.dot(CJ, T)
            T = np.dot(CK, T)
            T = np.dot(CL, T)
            #TOld = np.einsum("i,j,k,l,ijkl",
            #                 CI, CJ, CK, CL, self.eri_w.np,
            #                 optimize=True)
            Ex += BK*0.5*T
            return Ex

# Main ensemble class
# Does DFA part and J, K (EH, ExHF) energies
class EnsembleHelper:
    def __init__(self, wfn, JKFit=True, RKS=True,
                 Degen=False):
        self.wfn = wfn
        self.VPot = wfn.V_potential() # Note, this is a VBase class
        try:
            self.DFA = self.VPot.functional()
        except:
            self.DFA = None
            self.VPot = None

        SRI = wfn.S() # Overlap integral
        self.S = SRI.to_array(dense=True)
        SRI.power(-0.5, 1e-14)
        self.SRI = SRI.to_array(dense=True)
        
        if Degen:
            # Eventually needs to preserve symmetry of HOMO/LUMO
            F = wfn.Fa().to_array(dense=True)
            H0 = self.SRI.dot(F.dot(self.SRI))
            self.DegenHelp = DegenHelper(H=H0)
            eps_Sym, _ = self.DegenHelp.Solve(H0, Sort="Assign")
            eps = wfn.epsilon_a().to_array(dense=True)

            print("DEBUG: eps (psi4) - eps (sym) = ", eps-eps_Sym)
        else:
            self.DegenHelp = None

            
        self.nbf = self.wfn.nmo()
        self.NDOcc = wfn.nalpha()

        self.RKS = RKS
        if self.RKS:
            self.DM = psi4.core.Matrix(self.nbf, self.nbf)
            self.VM = psi4.core.Matrix(self.nbf, self.nbf)
        else:
            self.DMa = psi4.core.Matrix(self.nbf, self.nbf)
            self.DMb = psi4.core.Matrix(self.nbf, self.nbf)
            self.VMa = psi4.core.Matrix(self.nbf, self.nbf)
            self.VMb = psi4.core.Matrix(self.nbf, self.nbf)
        self.H_ao = self.wfn.H().to_array(dense=True)

        self.alpha = None
        self.omega = None
        if not(self.DFA is None):
            if self.DFA.is_x_hybrid():
                self.alpha = self.DFA.x_alpha()
                if self.DFA.is_x_lrc():
                    self.omega = self.DFA.x_omega()

        self.ERIHelp = ERIHelper(wfn,
                                 JKFit=JKFit,
                                 alpha=self.alpha, omega=self.omega)
        self.E = None
        self.F = None
        
    def Report(self):
        if self.DFA is None:
            return "Hartree-Fock"
        
        if self.DFA.is_x_hybrid():
            if self.DFA.is_x_lrc():
                Str = "RS hybrid (omega = %.3f) @ %.3f"%(self.DFA.x_omega(), self.DFA.x_alpha())
            else:
                Str = "Hybrid @ %.3f"%(self.DFA.x_alpha())
        else:
            Str = "Pure DFA"
        return Str
        
    def GetDFA(self, Da=None, Db=None,
               return_F=False,
               HFHelp=None):
        RHF = True
        if Da is None: Da = self.wfn.Da().to_array(dense=True)
        if Db is None:
            D = 2.*Da
            Db = Da
        else:
            D = Da+Db

        # Needs to be modifed to work with UKS
        TK.StartTimer("ExcDFA")
        if self.RKS:
            self.DM.np[:,:] = D/2.
            if not(self.VPot is None):
                self.VPot.set_D([self.DM])
                self.VPot.compute_V([self.VM])
            VDFT = self.VM.to_array(dense=True)
        else:
            self.DMa.np[:,:] = Da
            self.DMb.np[:,:] = Db
            if not(self.VPot is None):
                self.VPot.set_D([self.DMa,self.DMb])
                self.VPot.compute_V([self.VMa,self.VMb])
            VDFT = self.VMa.to_array(dense=True)

        #eri = self.mints.aO_eri().np
        TK.EndTimer("ExcDFA")

        self.Eob = np.tensordot(self.H_ao, D)
        if not(self.VPot is None):
            self.Exc = self.VPot.quadrature_values()["FUNCTIONAL"]
            # TEMP
            #print("\# Exc = %.5f"%(self.Exc))
            # END TEMP
        else: self.Exc = 0.

        TK.StartTimer("ExHF")
        if HFHelp is None:
            self.J, self.EH   = self.ERIHelp.JEH(D)
            self.F, self.ExHF = 0.,0.
            if Db is None:
                self.K, self.ExHF = self.ERIHelp.KEx(D)
            else:
                Ka, ExHFa = self.ERIHelp.KEx(Da, alfactor=1.)
                Kb, ExHFb = self.ERIHelp.KEx(Db, alfactor=1.)
                self.K = 0.5*(Ka + Kb)
                self.ExHF = ExHFa + ExHFb
        else:
            self.J = HFHelp.J()
            self.EH = HFHelp.EH()
            self.K = HFHelp.K()
            self.ExHF = HFHelp.ExHF()
        TK.EndTimer("ExHF")
            
        self.Enn = self.wfn.molecule().nuclear_repulsion_energy()
        
        self.E = self.Eob + self.EH + self.Exc + self.ExHF + self.Enn

        if not(return_F):
            return self.E, {'Enn':self.Enn,
                            'Eob':self.Eob,
                            'EH':self.EH,
                            'Exc':self.Exc,
                            'ExHF':self.ExHF, }
        else:
            self.F = VDFT + self.J + self.K + self.H_ao

            return self.E, self.F

    def AddF(self, DF):
        self.F += DF
        return self.F
    
    def AddJ(self, D, factor=1.,
             Freeze=False):
        J1, EH1   = self.ERIHelp.JEH(D)

        if Freeze:
            return EH1*factor, J1*factor

        # Add to appropriate terms
        self.J  += J1*factor
        self.EH += EH1*factor

        self.F  += J1*factor
        self.E  += EH1*factor
        
        return self.E, self.F

    def AddK(self, D, alfactor=0.5, factor=None,
             Freeze=False):
        K1, Ex1   = self.ERIHelp.KEx(D, alfactor=alfactor, factor=factor)

        if Freeze:
            return Ex1, K1
        
        # Add to appropriate terms
        self.K  += K1
        self.ExHF += Ex1

        self.F  += K1
        self.E  += Ex1
        
        return self.E, self.F

    def Solve(self, F=None, CR=None):
        if F is None: F = self.F # Set to default
        if F is None: return None # No default

        if CR is None:
            H = self.SRI.dot(F.dot(self.SRI))
            if self.DegenHelp is None:
                w, v = la.eigh(H)
            else:
                w, v = self.DegenHelp.Solve(H)

            return w, self.SRI.dot(v)
        else:
            return CR.Solve(F)

class HFHelper:
    def __init__(self, C, NDOcc, ERIHelp, DkL=0, DkH=0):
        quit()
        self.State = 0

        self.C = C
 
        kH = NDOcc-1
        kL = NDOcc+DkL

        Indx = np.array(list(range(NDOcc))+[kL])
        NIndx = Indx.shape[0]
        N2Indx = NIndx*(NIndx+1)//2
        
        J2Indx = np.zeros((NIndx,NIndx))
        K2Indx = np.zeros((NIndx,NIndx))
        for I in range(len(Indx)):
            for J in range(I,len(Indx)):
                CI = self.C[:,Indx[I]]
                CJ = self.C[:,Indx[J]]
                J2Indx[I,J] = ERIHelp.E_IJKL(CI,CI,CJ,CJ)
                K2Indx[I,J] = ERIHelp.E_IJKL(CI,CJ,CI,CJ)

                if not(I==J):
                    J2Indx[J,I] = J2Indx[I,J]
                    K2Indx[J,I] = K2Indx[I,J]

        self.EH=[0,0,0]
        self.ExHF=[0,0,0]
        self.J=[None,None,None]
        self.K=[None,None,None]

        EHCC = np.sum(J2Indx[:kH,:kH])
        ExCC = np.sum(K2Indx[:kH,:kH])

        EHCh = 2.*np.sum(J2Indx[:kH,kH])
        ExCh = 2.*np.sum(J2Indx[:kH,kH])

        EHCl = 2.*np.sum(J2Indx[:kH,kH+1])
        ExCl = 2.*np.sum(J2Indx[:kH,kH+1])

        EHhh = J2Indx[kH,kH]
        EHhl = 2.*J2Indx[kH,kH+1]
        EHll = J2Indx[kH+1,kH+1]

        Exhh = K2Indx[kH,kH]
        Exhl = 2.*K2Indx[kH,kH+1]
        Exll = K2Indx[kH+1,kH+1]

        self.EH  [0] = 4.*(EHCC + EHCh + EHhh)
        self.ExHF[0] =-2.*(ExCC + ExCh + Exhh)

        self.EH  [1] = (4.*EHCC + 2.*EHCh + 2.*EHCl + EHhh + EHhl + EHll)
        self.ExHF[1] =-(2.*ExCC + 2.*ExCh + 2.*ExCl + Exhh + Exhl + Exll)

        self.EH  [2] = 4.*(EHCC + EHCl + EHll)
        self.ExHF[2] =-2.*(ExCC + ExCl + Exll)

        
        
    def J(): return self.J[self.State]
    def K(): return self.K[self.State]
    def EH(): return self.EH[self.State]
    def ExHF(): return self.ExHF[self.State]
    
class OrbitalReduced:
    def __init__(self, EH, eps0, C0, NActive,
                 kL=None,
                 FreezeCore=0):
        NDOcc = EH.NDOcc
        NBF = C0.shape[0]

        self.eps0 = eps0*1.
        self.C0 = C0*1.

        if FreezeCore>NDOcc*2:
            print("Trying to freeze %d orbitals but only %d are occupied. Terminating."\
                  %(FreezeCore//2, NDOcc))
            quit()

        if NActive>0:
            self.HL = False
            self.k0 = max(NDOcc-NActive, FreezeCore//2)
            self.k1 = min(NDOcc+NActive, NBF)
            self.CR = C0[:,self.k0:self.k1]
        else:
            self.HL = True
            self.kH = NDOcc-1
            if kL is None: kL=NDOcc
            self.kL = kL
            self.CR = C0[:,[self.kH,self.kL]]
            
    def Solve(self, F):
        HR = (self.CR.T).dot(F.dot(self.CR))
        wr, vr = la.eigh(HR)

        w = self.eps0*1.
        C = self.C0*1.
        if not(self.HL):
            w[self.k0:self.k1] = wr
            C[:,self.k0:self.k1] = np.dot(self.CR, vr)
        else:
            w[self.kH] = w[0]
            w[self.kL] = w[1]
            C[:,self.kH] = np.dot(self.CR, vr[:,0])
            C[:,self.kL] = np.dot(self.CR, vr[:,1])
        return w, C

    
class EnsembleCalculator:
    def __init__(self, wfn, JKFit=True, RKS=True, Degen=False,
                 DkL=0, DkH=0, ExcAll=False,
    ):
        self.wfn = wfn
        self.EnsHelp = EnsembleHelper(wfn, JKFit=JKFit, RKS=RKS, Degen=Degen)
        self.NDOcc = wfn.nalpha()
        self.kH = max(0,self.NDOcc-1-DkH)
        self.kL = self.NDOcc+DkL

        self.ExcAll = ExcAll
        self.Ca0 = self.wfn.Ca().to_array(dense=True)
        self.epsilon0 = self.wfn.epsilon_a().to_array(dense=True)
        
        if DkL<0:
            ESSMin = 10000.
            for a in range(0,abs(DkL)):
                CH = self.Ca0[:,self.kH]
                CL = self.Ca0[:,self.kH+1+a]
                ESS = self.EnsHelp.ERIHelp.E_IJKL(CH, CL, CH, CL)
                print("kL = %3d, ESS = %.4f"%(self.kH+1+a, ESS))
                if (ESS<ESSMin):
                    self.kL = self.kH+1+a
                    ESSMin=ESS

        self.CalcDArr()

    def ShowF(self, F = None, epsilon=None):
        if F is None:
            E, F = self.EnsHelp.GetDFA(return_F=True)
        a = self.kH
        b = min(F.shape[0], self.kH+4)
        FC = F[a:b,a:b]
        Str = "F[kH:kH+4,kH:kH+4]=F[%d..%d,%d..%d] =\n"%(a,b,a,b)
        for k in range((b-a)):
            Str += "["+" ".join(["%8.4f"%(x) for x in FC[k,:]])+" ]\n"

        if not(epsilon is None):
            Str += "epsilon[kH;kH+4]=["
            Str += " ".join(["%8.4f"%(x) for x in epsilon[a:b]])
            Str += "]\n"
        return Str
        
    def ResetC(self):
        self.Ca = self.wfn.Ca()
    
    def CalcDArr(self, Ca=None):
        if Ca is None:
            self.Ca = self.wfn.Ca().np
        else:
            self.Ca = Ca
            
        self.CH = self.Ca[:,self.kH]
        self.CL = self.Ca[:,self.kL]
        if self.kH>0:
            self.CHm1 = self.Ca[:,self.kH-1]
        else: self.CHm1 = None

        # Occupation factos
        fu = np.zeros((3,self.Ca.shape[0],))
        fd = np.zeros((3,self.Ca.shape[0],))
        for k in range(3):
            fu[k,:self.NDOcc] = 1.
            fd[k,:self.NDOcc] = 1.
        # Singlet
        fu[1,self.kL]+=1.
        fd[1,self.kH]-=1.
        # Doublet
        fu[2,self.kL]+=1.
        fu[2,self.kH]-=1.
        fd[2,self.kL]+=1.
        fd[2,self.kH]-=1.

        self.fpArr = [(fu[0,:],fd[0,:]), (fu[1,:],fd[1,:]), (fu[2,:],fd[2,:])]
        self.fArr = [fu[0,:]+fd[0,:], fu[1,:]+fd[1,:], fu[2,:]+fd[2,:]]
        
        Dgs = 0.
        for k in range(self.NDOcc):
            Dgs += np.outer(self.Ca[:,k], self.Ca[:,k])

        DL = np.outer(self.CL, self.CL)
        DH = np.outer(self.CH, self.CH)

        self.DArr = [(Dgs,None), (Dgs+DL, Dgs-DH), (Dgs+DL-DH,None)]

        # Handle single-occupancy radicals
        if not(self.CHm1 is None):
            DHm1 = np.outer(self.CHm1, self.CHm1)
            Dpgs = Dgs-DH
            self.DpArr = [(Dgs,Dpgs), (Dgs, Dgs-DHm1), (Dpgs+DL,Dpgs)]
        else:
            self.DpArr = self.DArr
            
    def GetEn(self, w=[1.,0.,0.],
              UseHFHelper=False,
              SingletOnly=True,
              Doublet=False,
              Ansatz="FDT",
              return_D=False):

        # Singlet only ensemble
        if SingletOnly:
            wT = 0.
            if len(w)<3:
                wE = np.zeros((3,))
                wE[:len(w)]=w
            else:
                wE = np.array(w[:3])
        else:
            if len(w)<4:
                print("Must specify all weights for singlet/triplet ensembles")
                quit()
            else:
                wE = np.array([w[0],w[1]+w[2],w[3]])
                wT = w[1]
                
        EE = 0.*wE
        EMiscE = [None]*len(wE)

        if UseHFHelper:
            HFHelp = HFHelper(self.Ca, self.NDOcc, self.EnsHelp.ERIHelp)
            
        # Compute the average occupancy
        fw = 0.
        for k in range(len(wE)):
            fw += wE[k]*self.fArr[k]

        if not(Doublet):
            DArr = self.DArr
        else:
            DArr = self.DpArr # array of "cation"
            
        # Compute the average density matrix
        DAvg = 0.*DArr[0][0]
        for k in range(len(wE)):
            if not(DArr[k][1] is None):
                DAvg += wE[k]*(DArr[k][0] + DArr[k][1])
            else:
                DAvg += wE[k]*(2.*DArr[k][0])
        
        # Compute the elements
        for k in range(len(wE)):
            EE[k], EMiscE[k]\
                = self.EnsHelp.GetDFA(Da=DArr[k][0], Db=DArr[k][1])
        ESS = self.EnsHelp.ERIHelp.E_IJKL(self.CH, self.CL,
                                          self.CH, self.CL)*4.

        # Nuclear energy
        Enn = EMiscE[0]['Enn']
        
        # T+ VExt
        Eob = 0.
        for k in range(len(wE)):  Eob += wE[k]*EMiscE[k]['Eob']


        if Ansatz.upper()=="FDT":
            # Exc (two variants)
            Exc = 0.
            if self.ExcAll:
                for k in range(len(wE)):  Exc += wE[k]*EMiscE[k]['Exc']
            else:
                Exc += (wE[0]-wE[2])*EMiscE[0]['Exc'] \
                    + (wE[1]+2.*wE[2])*EMiscE[1]['Exc']

            # EH (using FDT)
            EH = 0.
            for k in range(len(wE)):  EH += wE[k]*EMiscE[k]['EH']
            EH += (wE[1]+wE[2]-wT)*ESS

            # ExHF (using FDT)
            ExHF = (wE[0]-wE[2])*EMiscE[0]['ExHF'] \
                + (wE[1]+2.*wE[2])*EMiscE[1]['ExHF']
        elif Ansatz.upper()[:3]=="ANS":
            if Ansatz[-1]=="1": # Ans1
                # Calculation on the average density matrix
                EAvg, EMiscAvg\
                    = self.EnsHelp.GetDFA(Da=DAvg)
                EH = EMiscAvg['EH']
                Exc = EMiscAvg['Exc']
                ExHF = EMiscAvg['ExHF']
            elif Ansatz[-1]=="2": # Ans1
                # Average - need to recompute ESS
                ESS = self.EnsHelp.ERIHelp.Ex_IJKL(self.CH, self.CL,
                                                  self.CH, self.CL)*4.

                Exc = 0.
                for k in range(len(wE)):  Exc += wE[k]*EMiscE[k]['Exc']
                
                # EH (using FDT)
                EH = 0.
                for k in range(len(wE)):  EH += wE[k]*EMiscE[k]['EH']

                # ExHF (using FDT)
                ExHF = 0.
                for k in range(len(wE)):  ExHF += wE[k]*EMiscE[k]['ExHF']
                ExHF += (wE[1]-wT)*ESS
            else:
                quit()
                
        # Total
        E = Eob + EH + Exc + ExHF + Enn

        RetMisc = {'Enn':Enn, 'Eob':Eob, 'EH':EH, 'Exc':Exc, 'ExHF': ExHF,
                   'ESS':ESS,
                   'fw':fw }
        if return_D: RetMisc['D']=DAvg
        return E, RetMisc
        
    def OptEn(self, w=[1.,0.,0.],
              SingletOnly=True,
              Doublet=False,
              Ansatz="FDT",
              F1=None, F2=None,
              SSOEP=False,
              a_0 = -0.5,  a_1 =  0.5,
              ECut = 5e-5, aCut = 1e-3,
              Report=False,
              NActive=None, FreezeCore=0,
              return_D=False,
              return_F=True,
    ):
        if F1 is None:
            self.ResetC()
            self.CalcDArr()
            E1, F1 = self.EnsHelp.GetDFA(Da=self.DArr[0][0],
                                         return_F=True)
            #E2, F2 = self.EnsHelp.GetDFA(Da=self.DArr[2][0], Db=self.DArr[2][1],
            #                             return_F=True)
            E2, F2 = self.EnsHelp.GetDFA(Da=self.DArr[2][0],
                                         return_F=True)

        # Initial energy with F1
        eps0, C0 = self.EnsHelp.Solve(F1)
        self.CalcDArr(C0)
        E00, Opt0 = self.GetEn(w,
                               SingletOnly=SingletOnly,
                               Doublet=Doublet,
                               Ansatz=Ansatz,
                               return_D=return_D)

        fw = Opt0['fw']
        if SSOEP:
            #print("fw = " + " ".join(["%5.3f"%(x) for x in fw]))
            V = GetVStep(self.EnsHelp.wfn, C=C0, fw=fw)
            #print(F1[:7,:7])
            #print(V[:7,:7])
            F2 = F1 - V
            #quit()
            
        if NActive is None and FreezeCore==0:
            CR = None
        else:
            if NActive is None: NActive = C0.shape[0]
            CR = OrbitalReduced(self.EnsHelp, eps0, C0,
                                NActive=NActive, kL=self.kL,
                                FreezeCore=FreezeCore)
            eps, C = self.EnsHelp.Solve(F1, CR)


        # Iterate to the optimal solution
        r1 = (np.sqrt(5.)-1.)/2.
        r2 = r1**2

        a0 = a_0 + (a_1-a_0)*np.array([0,r2,r1,1.])
        E0 = 0.*a0
        for k in range(4):
            a = a0[k]
            TK.StartTimer("Solve")
            eps, C = self.EnsHelp.Solve((1.-a)*F1 + a*F2, CR)
            TK.EndTimer("Solve")

            self.CalcDArr(C)
            TK.StartTimer("GetEn")
            E0[k], _ = self.GetEn(w,
                                  SingletOnly=SingletOnly,
                                  Doublet=Doublet,
                                  Ansatz=Ansatz,
            )
            TK.EndTimer("GetEn")
            
        NEval =4

        aFull = np.zeros(50)
        EFull = np.zeros(50)
        aFull[:4] = a0
        EFull[:4] = E0
        kFull = 4

        while (E0.max()-E0.min()>ECut) and (a0[3]-a0[0]>aCut):
            kk = E0.argmin()
            if kk<2:
                anew = a0[0] + (a0[2]-a0[0])*r2
            else:
                anew = a0[1] + (a0[3]-a0[1])*r1
            
            TK.StartTimer("Solve")
            eps, C = self.EnsHelp.Solve((1.-anew)*F1 + anew*F2, CR)
            TK.EndTimer("Solve")
            self.CalcDArr(C)
            TK.StartTimer("GetEn")
            Enew, _ = self.GetEn(w,
                                 SingletOnly=SingletOnly,
                                 Doublet=Doublet,
                                 Ansatz=Ansatz,
            )
            TK.EndTimer("GetEn")
            NEval+=1

            aFull[kFull] = anew
            EFull[kFull] = Enew
            kFull +=1
            
            if kk<2:
                a0[:] = [a0[0],anew,a0[1],a0[2]]
                E0[:] = [E0[0],Enew,E0[1],E0[2]]
            else:
                a0[:] = [a0[1],a0[2],anew,a0[3]]
                E0[:] = [E0[1],E0[2],Enew,E0[3]]

        
                
        kk = E0.argmin()
        aFinal = a0[kk]
        EFinal = E0[kk]

        # Reset if energy is greated than the original value E00
        if EFinal>(E00+1e-4):
            print("Something has gone wrong:")
            print("DE(min) = %6.4f, E(min) = %12.4f found for a(min) = %.4f [%2d calculations]"\
                  %(EFinal-E00, EFinal, aFinal, NEval))
            aFinal = 0.
            EFinal = E00

        # If it is at an edge show some information
        if (aFinal == a_0) or (aFinal == a_1):
            print("# Warning - edge case aF = %.3f in (%.3f, %.3f)"%(aFinal, a_0, a_1))
            print("a0 = %s, E0 = %s"%( ", ".join(["%.3f"%(x) for x in a0]),
                                       ", ".join(["%.3f"%(x) for x in E0])))

        # Reevaluate
        FFinal = (1.-aFinal)*F1 + aFinal*F2
        epsFinal, C = self.EnsHelp.Solve(FFinal, CR)
        self.CalcDArr(C)
        EFinal, EMiscFinal = self.GetEn(w,
                                        SingletOnly=SingletOnly,
                                        Doublet=Doublet,
                                        Ansatz=Ansatz,
                                        return_D=return_D
        )
        NEval+=1


        OptMisc =  {
            'E00': E00,
            'aFull':aFull, 'EFull':EFull,
            'a0':a0, 'E0':E0,
            'aFinal':aFinal, 'EFinal':EFinal,
            'FFinal':FFinal, 'epsFinal':epsFinal,
        }

        if Report:
            print("DE(min) = %6.4f, E(min) = %12.4f found for a(min) = %.4f [%2d calculations]"\
                  %(EFinal-E00, EFinal, aFinal, NEval))

        if return_F: OptMisc['FFinal']=(1.-aFinal)*F1 + aFinal*F2
        if return_D: OptMisc['D0']=Opt0['D']
        
        return EFinal, EMiscFinal, OptMisc


def MaskToWeights(w, Mask):
    M = np.array(Mask)
    IL = np.argwhere(M<0.)
    IG = np.argwhere(M>0.)

    WP = w*np.sum(M[IG])
    WL = (1.-WP)/len(IL)

    ww = np.zeros(M.shape)
    ww[IL] = WL
    ww[IG] = w*M[IG]

    return ww


if __name__ == "__main__":
    1
