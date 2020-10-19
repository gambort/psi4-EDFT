import time

TimeKeeper_All = {}

def StartTimer(ID):
    if not(ID) in TimeKeeper_All:
        TimeKeeper_All[ID] = [0.,0.]

    TimeKeeper_All[ID][0]=time.time()

def EndTimer(ID):
    if not(ID) in TimeKeeper_All:
        print("Warning! %s not initiated - results will be nonsense"%(ID))

    DT = time.time() - TimeKeeper_All[ID][0]

    TimeKeeper_All[ID][1] += DT

def ShowTimer(ID):
    T = abs(TimeKeeper_All[ID][1])
    if T<300.: TStr = "%10.0fs"%(T)
    elif T<18000.: TStr = "%10.0fm"%(T/60.)
    else: TStr = "%10.2fhr"%(T/3600.)
    print("\# Time spent in %10s = %s mins"%(ID, TStr))


def ShowTimers():
    for ID in TimeKeeper_All:
        ShowTimer(ID)


