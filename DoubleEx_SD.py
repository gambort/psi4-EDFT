#!/home/timgould/psi4conda/bin/python

HomeDir = "/home/578/tjg578/Molecules/EDFT-2"

import sys
import os
sys.path.append(HomeDir)

import psi4
import numpy as np

from MoleculeDefs import *

import psi4EDFT.TimeKeeper as TK
from psi4EDFT.LibEnsemble import *
from psi4EDFT.LibDFAs import *
from psi4EDFT.LibEDFTCalculator import *

# See:
# http://forum.psicode.org/t/basic-psi4numpy-question-for-the-dft-potential/661/2
# http://www.psicode.org/psi4manual/master/api/psi4.core.VBase.html
# http://www.psicode.org/psi4manual/master/api/psi4.core.LibXCFunctional.html
# http://forum.psicode.org/t/obtain-the-energy-from-dft-at-arbitrary-density-matrix/1884/4

DefaultMol = "LiH_3.5"
DefaultDFA = "PBE"
DefaultBasis = 'def2-tzvp'

from  optparse import OptionParser

parser = OptionParser()
parser.add_option('-M', type="string", dest="Mol", default=DefaultMol,
                  help="""Specify the molecule file name -- some atoms
and diatoma can be specified using, e.g., Be or
LiH_3.5 for Li-H with 3.5A separation""")
parser.add_option('--Molecule', type="string", dest="Mol", default=DefaultMol,
                  help="""Longer form of -M""")
parser.add_option('--DFA', type="string", default=DefaultDFA,
                  help="""Specify any DFA implemented in Psi4""")
parser.add_option('--Basis', type="string", default=DefaultBasis,
                  help="""Specify any basis with a corresponding
[basis]-JKFit counterpart""")

parser.add_option('--UseDegen', dest="Degen", action="store_true",
                  default=False,
                  help="""Use a degenerate solver (expert mode
only and not recommended)""")

parser.add_option('--Coarse', action="store_true", default=True,
                  help="""Use a coarser grid""")
parser.add_option('--Fine', dest="Coarse", action="store_false",
                  help="""Use a finer grid""")

parser.add_option('--Ansatz', type="string", default="FDT",
                  help="""Specficy the EDFT ansatz (expert mode)""")
parser.add_option('--SDOnly', action="store_true", default=False,
                  help="""Compute the state-driven energy only, i.e.
do EDFT using ground state KS orbitals""")
parser.add_option('--SaveDM', type="string", default=None,
                  help="""Save the density matrices to this file""")


parser.add_option('--SSOEP', action="store_true", default=False,
                  help="""Use the SSOEP (expert only,
not recommended)""")
parser.add_option('--UKS', dest="RKS", action="store_false", default=False,
                  help="""Do unconstrainted KS DFT""")
parser.add_option('--RKS', dest="RKS", action="store_true", default=False,
                  help="""Do restricted KS DFT""")
parser.add_option('--ExcAll', dest="ExcAll",
                  action="store_true", default=False,
                  help="""NOT USED""")

parser.add_option('--DkH', type="int", default=0,
                  help="""Specify effective HOMO = HOMO - [this value] to
deal with unusual KS ordering""")
parser.add_option('--DkL', type="int", default=0,
                  help="""Specify effective LUMO = HOMO + 1 + [this value] to
deal with unusual KS ordering""")
parser.add_option('--Nw', type="int", default=5,
                  help="""Number points for quadratic fit""")

parser.add_option('--NActive', type="int", default=None,
                  help="""Use a restricted active space (not recommended)""")
parser.add_option('--FreezeCore', type="int", default=0,
                  help="""Freeze some core orbitals (not recommended)""")

(Opts, args) = parser.parse_args()

Mol = Opts.Mol
Basis = Opts.Basis
DFA, DFATxt = GetDFA(Opts.DFA)

JKFit = True

print("="*72)
print("""
   EEEEE DDDD  FFFFF TTTTT
   E     D   D F       T  
   EEEEE D   D FFFF    T    @ PSI4
   E     D   D F       T  
   EEEEE DDDD  F       T  
""")
print("="*72)

psi4.core.set_output_file('__Psi4Output.dat', True)
np.set_printoptions(precision=5, suppress=True)

if 'MKL_NUM_THREADS' in os.environ:
    NThreads = int(os.environ['MKL_NUM_THREADS'])
else: NThreads = 4

psi4.core.set_num_threads(NThreads)
try:
    import mkl
    mkl.set_num_threads(NThreads)
except:
    pass
print("Number of threads = %d"%(psi4.core.get_num_threads()))

def GetMolStr(Mol, Sz=1):
    MolStr = GetMolecule(Mol, Sz=Sz)
    if MolStr is None:
        MolStr = ReadMolecule(Mol)
        T = MolStr.split()
        Q = int(T[0])
        MolStr="%d %d\n"%(Q,Sz) + MolStr[4:]
    return MolStr
    
print("DFA = %10s, Basis = %12s"%(DFATxt, Basis))
print(GetMolStr(Mol, Sz=1))

psi4.set_options({
    #'guess': 'SAD',
    'basis_guess':'3-21G',
    'maxiter': 200,
    'd_convergence': 1e-5,
    'e_convergence': 1e-6,
})

if Opts.Coarse:
    print("Using a coarse grid")
    psi4.set_options({
        'dft_spherical_points': 110,
        'dft_radial_points': 50,
    })
else:
    print("Using default (fine) grid")

mol = psi4.geometry(GetMolStr(Mol, Sz=3))
psi4.set_options({
    'reference':'uhf',
    'basis':Basis,
})
try:
    Ets_scf, wfnts = psi4.energy("scf", dft_functional = DFA, return_wfn=True)
except: Ets_scf = 0.

SystemStr = GetMolStr(Mol, Sz=1)
mol = psi4.geometry(SystemStr)
psi4.set_options({
    'reference':{True:'rhf', False:'uhf'}[Opts.RKS],
    'basis':Basis,
})
E_scf, wfn = psi4.energy("scf", dft_functional = DFA, return_wfn=True)


print("="*72)
print("E = %.5f E_ts = %.5f"%(E_scf, Ets_scf))
if not(Ets_scf==0.):
    print("DFT Triplet excitation: %5.2f"%(eV*(Ets_scf-E_scf)))

print("Energy(S0) = %12.2f [eV]"%(eV*E_scf))
EDFTOptions = EDFTOptionsClass()
EDFTOptions.ParseOpts(Opts)
EDFTOptions.Report()

EDFTCalculator(wfn, EDFTOptions,
               System=SystemStr,
               SSOEP=Opts.SSOEP)
print("="*72)
TK.ShowTimers()
print("="*72)
