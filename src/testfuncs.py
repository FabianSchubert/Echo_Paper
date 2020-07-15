#!/usr/bin/env python3

import numpy as np

def gen_in_out_one_in_subs(T,tau):
    inp = np.ndarray((T))

    inp = ((np.random.rand(T) < .5)*1. -.5)*2.

    outp = np.ndarray((T))

    for k in range(T):

        if k - tau - 1 < 0:
            outp[k] = (np.random.rand() < .5)*1.
        else:
            outp[k] = (inp[k-tau] != inp[k-tau-1])*1.

    outp = (outp - .5)*2.

    return inp, outp
