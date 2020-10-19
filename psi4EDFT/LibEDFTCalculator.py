#!/home/timgould/psi4conda/bin/python

LibEDFTCalculator_Info=\
"""
Routines to run EDFT calculations after an initial ground state
calculation. The front end to this library is via:

Results =  EDFTCalculator(wfn, Opts, System="information")

where Opts is an EDFTOptionsClass containing information about
the calculation. Many options are technical. Important ones are:

RKS - True/False
      True:  do calculations with restricted orbitals and
             spin-independent DFAs (even for triple states)
      False: do calculations with untrstricted orbitals and DFA
             starting from an UKS ground state calculation
SDOnly - True/False
      Stop after the EDFT@KS level, i.e. do no self-consistency
SaveDM - Filename
      If not None store the density matrix information in the
      specified output file
DkH - integer [default 0]
      Set effective HOMO = HOMO - DkH to access, e.g., a different
      symmetry state (this is imperfect)
DkL - integer [default 0]
      Set effective LUMO = HOMO + 1 + DkL to access, e.g., a different
      symmetry state (this is imperfect)
"""


import psi4
import numpy as np

from psi4EDFT.LibEnsemble import *

eV = 27.211
ha = 0.5
th = 1./3.

class EDFTOptionsClass(dict):
    def __init__(self,
                 DFA="PBE", # Default DFA
                 Basis="def2-tzvp", # Default basis
                 Ansatz="FDT", # Default ansatz
                 JKFit=True, # Use JK density fitting
                 Degen=False, # Preserve degeneracies
                 SDOnly=False, # Only do the SD term
                 Nw=5, # Number of weights for quadratic fit
                 NActive=None, FreezeCore=0, # For acceleration
                 DkH=0, # HOMO = HOMO - DkL
                 DkL=0, # LUMO = HOMO + 1 + DkL
                 RKS=True, # Use restricted KS calculations
                 ExcAll=False, # Use alternative Exc expression
                 SaveDM=None, # Save the density matrix
                 Opts=None, # Options to be parsed
                 ):
        super().__init__(self)
        self.__dict__ = self

        # Get directly
        self.DFA = DFA
        self.Basis = Basis

        self.Ansatz = Ansatz
        
        self.JKFit = JKFit
        self.Degen = Degen
        
        self.SDOnly = SDOnly
        self.Nw = Nw
        self.NActive = NActive
        self.FreezeCore = FreezeCore

        self.DkH = DkH
        self.DkL = DkL
        self.RKS = RKS
        self.ExcAll = ExcAll

        self.SaveDM = SaveDM
        
        if not(Opts is None): self.ParseOpts(Opts)
    
    def ParseOpts(self, Opts):
        Opts_Dict = {key:value for key, value in Opts.__dict__.items() \
                     if not key.startswith('__') and not callable(key)}
        # Overwrite existing values using ID
        for ID in dict(self):
            if ID in Opts_Dict: self[ID] = Opts_Dict[ID]

        return self

    def Report(self):
        print("EDFT options: SDOnly = %s, JKFit = %s, NActive = %s, FreezeCore = %d"\
              %(self.SDOnly, self.JKFit, self.NActive, self.FreezeCore))
        return self
        
def EDFTCalculator(wfn, Opts,
                   System="Unknown",
                   SSOEP=False):
    EnsCalc = EnsembleCalculator(wfn,
                                 JKFit=Opts.JKFit,
                                 Degen=Opts.Degen,
                                 DkH=Opts.DkH,
                                 DkL=Opts.DkL,
                                 RKS=Opts.RKS,
                                 ExcAll=Opts.ExcAll)

    CalcData = {'System':System,
                'DkH':Opts.DkH,
                'DkL':Opts.DkL,
                'DFA':Opts.DFA,
                'Basis':Opts.Basis,
    }
    
    SDProps = EDFTSDCalculator(EnsCalc, Opts)
    CalcData['SD'] = SDProps

    if not(Opts.SDOnly):
        DDProps = EDFTDDCalculator(EnsCalc, Opts, SSOEP=SSOEP)
    else: DDProps = {}
    CalcData['DD'] = DDProps

    # Save the density matrix if asked
    if not(Opts.SaveDM is None):
        D0 = SDProps['D0']
        
        if not('DT' in DDProps): DT = SDProps['DT']
        else: DT = DDProps['DT']

        if not('D1' in DDProps): D1 = SDProps['D1']
        else: D1 = DDProps['D1']

        if not('D2' in DDProps): D2 = SDProps['D2']
        else: D2 = DDProps['D2']
        
        CalcData['DM'] = {'D0':D0, 'DT':DT, 'D1':D1, 'D2':D2}

        if Opts.SaveDM[-3:].upper()=="NPZ":
            np.savez(Opts.SaveDM, CalcData=CalcData)

    return CalcData

def EDFTSDCalculator(EnsCalc, Opts):
    # Do we need the density matrix
    return_D = not(Opts.SaveDM is None)
    
    E0, Misc0 = EnsCalc.GetEn(w = [1.,0.,0.],
                              Ansatz=Opts.Ansatz,
                              return_D = return_D)
    ET, _ = EnsCalc.GetEn(w = [0.,1.,0.,0.],
                          Ansatz=Opts.Ansatz,
                          SingletOnly=False)
    EA, _ = EnsCalc.GetEn(w = [ha,ha,0.],
                          Ansatz=Opts.Ansatz)
    EB, _ = EnsCalc.GetEn(w = [th,th,th],
                          Ansatz=Opts.Ansatz)
    E1 = 2.*EA - E0
    E2 = 3.*EB - E0 - E1

    Ehl = Misc0['ESS']/4.

    print("="*72)
    print(EnsCalc.ShowF(epsilon=EnsCalc.epsilon0))
    print("State-dependent only")
    print("E0 = %.4f E1 = %.4f E1(TS) = %.4f E2 = %.4f [Ha]"%(E0,E1,ET,E2))
    print("Ehl(1) = %6.2f Ehl(2) = %6.2f [eV]"%(Ehl*eV, Ehl*eV))
    print("SD:Delta Tx = %6.2f [eV]"%((ET-E0)*eV))
    print("SD:Delta Sx = %6.2f [eV]"%((E1-E0)*eV))
    print("SD:Delta Dx = %6.2f [eV]"%((E2-E0)*eV))

    RetDict = {'E0':E0, 'ET':ET, 'E1':E1, 'E2':E2, 'Ehl':Ehl}
    if return_D:
        RetDict['D0'] = EnsCalc.DArr[0][0]*2.
        RetDict['D1'] = EnsCalc.DArr[1][0]+EnsCalc.DArr[1][1]
        RetDict['DT'] = RetDict['D1']
        RetDict['D2'] = EnsCalc.DArr[2][0]*2.

    return RetDict

def DoLoop(EnsCalc, w, Mask,
           Ansatz="FDT",
           SSOEP = False,
           ID = "xx",
           SingletOnly=False,
           return_D=False,
           NActive=None, FreezeCore=0):

    Ew = 0.*w
    Ew0 = 0.*w
    ESSw = 0.*w
    Dw = [None]*len(w)
    
    for k,wk in enumerate(w):
        wSet = MaskToWeights(wk, Mask)

        Ew[k], Misc, Opt = EnsCalc.OptEn(w = wSet,
                                         Ansatz=Ansatz,
                                         SSOEP = SSOEP,
                                         SingletOnly=SingletOnly,
                                         return_D = return_D,
                                         NActive=NActive,
                                         FreezeCore=FreezeCore)
        Ew0[k] = Opt['E00']
        ESSw[k] = Misc['ESS']
        if return_D: Dw[k] = Misc['D']
        
        print("%-4s w = %.3f E = %8.2f [%8.2f]"%(ID, wk, eV*(Ew[k]-Ew[0]), eV*(Ew0[k]-Ew0[0])))

    if len(w)>=3:
        p = np.polyfit(w, Ew, 2)
        E0 = np.polyval(p, 0.)
        EX = np.polyval(p, 1.)

        Ehl = np.polyval(np.polyfit(w, ESSw, 2), 1.)/4.
    
        p0 = np.polyfit(w, Ew0, 2)
        E00 = np.polyval(p0, 0.)
        EX0 = np.polyval(p0, 1.)
    elif len(w)==2:
        if np.abs(w[0])>0.:
            print("Must include zero for Nw=2")
            quit()
            
        h = w[1]
        G = (Ew0[1]-Ew0[0])/h
        v = (Ew[1]-Ew[0])

        E0  = Ew[0]
        E00 = Ew0[0]
        EX  = Ew[0] + G + (v-G*h)/h**2
        EX0 = Ew0[0] + G

        Ehl = np.polyval(np.polyfit(w, ESSw, 1), 1.)/4.
        
    print("%-4s w = %.3f E = %8.2f [%8.2f] -- %8.2f"\
          %(ID, 1., eV*(EX-E0), eV*(EX0-E00), eV*(Ew[0]-E0)))

    if eV*np.abs(Ew[0]-E0)>0.05:
        print("Warning: E0(fit) and E0 differ by %8.2f eV"%(eV*(Ew[0]-E0)))

    MiscDict = {'w':w, 'Ew':Ew,}

    print(EnsCalc.ShowF(Opt['FFinal'], Opt['epsFinal']))

    if return_D:
        NB = Dw[0].shape[0]
        DOut = np.zeros((NB,NB))
        Dfit = np.zeros((len(w),NB))
        for k in range(NB):
            for i in range(len(w)):
                Dfit[i,:] = Dw[i][:,k]
            p = np.polyfit(w, Dfit, 2)
            DOut[:,k] = np.polyval(p, 1.)

        MiscDict['D'] = DOut
    
    return EX, EX0, Ehl, MiscDict


def EDFTDDCalculator(EnsCalc, Opts, SSOEP=False):
    # Do we need the density matrix
    return_D = not(Opts.SaveDM is None)

    E0, Misc0, Opt0 = EnsCalc.OptEn(w = [1.,0.,0.],
                                    Ansatz=Opts.Ansatz,
                                    NActive=Opts.NActive,
                                    FreezeCore=Opts.FreezeCore,
                                    return_D=return_D)

    Nw = Opts.Nw

    print("="*72)
    print(EnsCalc.ShowF())
    if SSOEP:
        print("Model = SSOEP")
    else:
        print("Model = Operator")
    print("="*72)

    # Triplet only
    wT = np.linspace(0.,ha,Nw)
    ET, ET0, EhlT, MiscT = DoLoop(EnsCalc, wT, [-1,1,0,0], ID="tx",
                                  SSOEP = SSOEP,
                                  Ansatz = Opts.Ansatz,
                                  SingletOnly=False,
                                  return_D = return_D,
                                  NActive=Opts.NActive,
                                  FreezeCore=Opts.FreezeCore)
    
    # Singlet only
    wA = np.linspace(0.,ha,Nw)
    EA, EA0, EhlA, MiscA = DoLoop(EnsCalc, wA, [-1, 1, 0], ID="sx",
                                  SSOEP = SSOEP,
                                  Ansatz = Opts.Ansatz,
                                  return_D = return_D,
                                  SingletOnly=True,
                                  NActive=Opts.NActive,
                                  FreezeCore=Opts.FreezeCore)

    # Doublet too
    wB = np.linspace(0.,th,Nw)
    EB, EB0, EhlB, MiscB = DoLoop(EnsCalc, wB, [-1,-1, 1], ID="dx",
                                  SSOEP = SSOEP,
                                  Ansatz = Opts.Ansatz,
                                  SingletOnly=True,
                                  return_D = return_D,
                                  NActive=Opts.NActive,
                                  FreezeCore=Opts.FreezeCore)


    E1 = EA
    E2 = EB

    print("="*72)
    print("All correlations")
    print("E0 = %.4f E1 = %.4f E1(TS) = %.4f E2 = %.4f [Ha]"%(E0,E1,ET,E2))
    print("Ehl(1) = %6.2f Ehl(2) = %6.2f [eV]"%(EhlA*eV, EhlB*eV))
    print("DD:Delta Tx = %6.2f [eV]"%((ET-E0)*eV))
    print("DD:Delta Sx = %6.2f [eV]"%((E1-E0)*eV))
    print("DD:Delta Dx = %6.2f [eV]"%((E2-E0)*eV))

    RetDict = {'E0':E0, 'ET':ET, 'E1':E1, 'E2':E2,
               'EhlT':EhlT, 'Ehl1':EhlA, 'Ehl2':EhlB}
    if return_D:
        RetDict['DT'] = MiscT['D']
        RetDict['D1'] = MiscA['D']
        RetDict['D2'] = MiscB['D']

    return RetDict
