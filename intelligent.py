# Author: Jose G. Perez <jperez50@miners.utep.edu>
# Hurwicz Criterion Experiments
# Requires sympy and numpy libraries

from sympy import Interval, Union
import numpy as np
np.random.seed(1738)

def H(S, alpha):
    S = list(S.atoms())
    a = S[1]
    b = S[0]
    return (alpha * a) + ((1 - alpha) * b)

# for alph in np.arange(0.1, 0.9, 0.1):
#%%
alph = 0.8
for i in range(15):
    s1_min = np.random.randint(-100,100)
    s1_max = np.random.randint(s1_min,100)
    s2_min = np.random.randint(-100,100)
    s2_max = np.random.randint(s2_min,100)

    S1 = Interval(s1_min, s1_max)
    S2 = Interval(s2_min, s2_max)
    SU = Union(S1, S2)
    HS1 = H(S1, alph)
    HS2 = H(S2, alph)
    HSU = H(SU, alph)
    print("=====Alpha={}".format(alph))
    print("S1={}, H[S1]={}".format(S1,HS1))
    print("S2={}, H[S2]={}".format(S2,HS2))
    print("SU={}, H[SU]={}".format(SU,HSU))

    if HSU == HS1:
        print("H[SU] is H[S1]")
    elif HSU == HS2:
        print("H[SU] is H[S2]")