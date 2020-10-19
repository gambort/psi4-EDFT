import psi4

def GetDFA(DFAID):
    if DFAID.upper() in ("HF", "SCF"):
        # Need to mix a tiny amount of PBE in to make
        # the code work
        eta = 1e-7
        DFATxt = "HF"
        DFA = {
            "name": DFATxt,
            "x_hf": {"alpha": 1.-eta},
            "x_functionals": {"GGA_X_PBE": {"alpha":eta}},
            "c_functionals": {"GGA_C_PBE": {"alpha":eta}},
        }
    elif DFAID.upper() in ("LC-PBE", "LCPBE"):
        DFATxt = "LC-PBE"
        DFA = {
            "name": DFATxt,
            "x_functionals": {
                "GGA_X_WPBEH": { "alpha":0.75, "omega":0.30 },
            },
            "c_functionals": { "GGA_C_PBE": {} },
            "x_hf": {
                "alpha": 0.25,
                "beta" : 1.00,
                "omega": 0.30 }
        }
    else:
        DFATxt = DFAID
        DFA = DFAID

    return DFA, DFATxt
