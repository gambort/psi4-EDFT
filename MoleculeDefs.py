import numpy as np

AtomList = [
    'He', 'Be', 'C', 'O', 'Ne',
    'Mg', 'Si', 'S', 'Ar', 'Ca',
    'Ti',
]

DiatomList = {'H2':('H', 'H', 1.0),  'HF':('H', 'F', 1.0),  'LiF':('Li', 'F', 1.0),  'LiH':('Li', 'H', 1.0), 
              'BH':('B', 'H', 1.0),  'CO':('C', 'O', 1.0), 
              'CNm':('C', 'N', 1.1, -1),  'O2':('O', 'O', 1.21), 
              'CrH':('Cr', 'H', 1.3), 
              'TiO':('Ti', 'O', 1.63),
              'CHp':('C', 'H', 1.0, 1), 
              'NF':('N', 'F', 1.32),  'NH':('N', 'H', 1.59), 
              'PF':('P', 'F', 1.32), 
              'PH':('P', 'H', 1.42), 
              'NOm':('N', 'O', 1.26, -1), 
              'S2':('S', 'S', 1.89), 
              'SO':('S', 'O', 1.48),
              'C2':('C', 'C', 2.36*0.53),
}

def ReadMolecule(FileName):
    F = open(FileName)
    if FileName[-3:].upper()=="XYZ":
        N = int(F.readline())
        F.readline()
        X = []
        for i in range(N):
            L = F.readline()
            X+=[L.rstrip()]
        mol = '0 1\n\n' + '\n'.join(X)
        mol += "\n\nsymmetry c1\n"
    else:
        mol = '\n'.join([L.rstrip() for L in F]) + '\n\n'
    F.close()
    return mol


def GetMolecule(Select, Sz=1):
    try:
        S = Select.index('_')
        ID = Select[:S]
        try:
            Val = float(Select[S + 1:])
        except:
            Val = Select[S + 1:]

        print('Getting %s with "%s"' % (ID, Val))
    except:
        ID = Select
        Val = None

    if ID in AtomList:
        if Val is not None:
            if Val.upper() == 'SYM':
                pass
            return '\n0 1\n\n%s\n' % ID
        return '\n0 %d\n\n%s\n\nsymmetry c1' % (Sz, ID)
    elif ID in DiatomList:
        if len(DiatomList[ID]) < 4:
            C = 0
            A1, A2, D = DiatomList[ID]
        else:
            A1, A2, D, C = DiatomList[ID]
        if Val is not None:
            D = Val
        return '\n%d %d\n%s\n%s 1 %.4f\n\nsymmetry c1' % (C, Sz, A1, A2, D)
    else:
        if Select[:3] == 'H2O':
            if len(Select) > 3:
                Ang = float(Select[4:]) / 180.0 * np.pi
            else:
                Ang = 110
            c = np.cos(Ang / 2.0)
            s = np.sin(Ang / 2.0)
            d = 0.958
            ca = c * d
            sa = s * d
            mol = '\n  0 1\nO 0.  0.0000  0.0000\nH 0. %7.4f %7.4f\n            H 0. %7.4f %7.4f\n            \n            symmetry c1\n' % (sa, -ca, -sa, -ca)
        elif Select[:4]=="CH3I":
            T = Select.split("_")
            if len(T)<2: D = 2.1576
            else: D = float(T[1])
            
            mol = "\n 0 %d\n"%(Sz)
            mol += """
I    0.0000    0.0000   %6.4f
C    0.0000    0.0000    0.0000
H    1.0139    0.2034   -0.3246
H   -0.6831    0.7764   -0.3246
H   -0.3308   -0.9798   -0.3246
"""%(D)
        else:
            mol = None
        return mol
