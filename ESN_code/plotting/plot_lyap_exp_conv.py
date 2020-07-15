#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

plt.rc('text.latex', preamble=r'''
\usepackage{dejavu}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}''')

from stdParams import *

import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hide_plot",action='store_true')

args = parser.parse_args()

from tqdm import tqdm


from scipy.linalg import logm

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

N = 500

W = np.random.normal(0.,1.,(N,N))
W[range(N),range(N)] = 0.

#optionally, randomize rows:
#W = (W.T * np.random.rand(N)).T

t = np.arange(1,30,5)
t = np.append(t,30)
nt = t.shape[0]

sigm_squ = np.ndarray((nt,N))

W /= np.abs(np.linalg.eigvals(W)).max()

for k in tqdm(range(nt)):
    T = t[k]

    W_pow = np.linalg.matrix_power(W,T)
    L = logm(W_pow.T @ W_pow)/(2.*T)
    sigm_squ_T = np.linalg.eigvals(L)
    sigm_squ_T = np.sort(sigm_squ_T.real)
    sigm_squ[k,:] = sigm_squ_T

l_analytic = np.sort(np.log(np.abs(np.linalg.eigvals(W))))

fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

for k,T in enumerate(t):
    col = np.array(colors[0])*(nt-1-k)/(nt-1) + np.array(colors[1])*k/(nt-1)
    if k==0 or k==nt-1:
        ax.plot(sigm_squ[k,:],'.',c=col,
        label="$n="+str(T)+"$",markersize=5)
    else:
        ax.plot(sigm_squ[k,:],'.',c=col,markersize=4)

ax.plot(l_analytic,'.',markersize=4,
        label='$\\ln\\left||\\lambda_i\\right||$',
        c=colors[2])

ax.legend()

ax.set_xlabel("$k$")
ax.set_ylabel("$k$th Eigenvalue")

ax.set_xlim([150,N+25])
ax.set_ylim([-1.,1.])

fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'lyap_exp_conv.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'lyap_exp_conv.png'),dpi=300)

if not(args.hide_plot):
    plt.show()
