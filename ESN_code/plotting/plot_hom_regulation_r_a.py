#!/usr/bin/env python3
import numpy as np
from stdParams import *
import os
import sys
import glob
from pathlib import Path

from src.analysis_tools import get_simfile_prop

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from tqdm import tqdm

import sys

import argparse

def plot(ax,input_type,adaptation_mode,col=colors[0]):

    file = get_simfile_prop(os.path.join(DATA_DIR,input_type
    +'_input_ESN/hom_regulation/hom_regulation_'+adaptation_mode))

    dat = np.load(file[0])

    a_rec = dat['a']
    N=dat['N']
    n_samples=dat['n_samples']
    W = dat['W']

    r_a = a_rec[0,:,:]**2. * (W[0,:,:]**2.).sum(axis=1)
    
    
    
    ax.plot(r_a[:,0],c=col,alpha=0.25,label='$R^2_{{\\rm a},i}$')
    ax.plot(r_a[:,1:100],c=col,alpha=0.25)

    ax.plot(r_a.mean(axis=1),'--',c='k',label='$R^2_{\\rm a}$',lw=2)

    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)

    leg = ax.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    ax.set_xlabel('time steps')
    ax.set_ylabel('$R^2_{{\\rm a},i}$')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("input_type",
    help='''specify four type of input (homogeneous_identical_binary,
    homogeneous_independent_gaussian, heterogeneous_identical_binary,
    heterogeneous_independent_gaussian)''',
    choices=['homogeneous_identical_binary',
    'homogeneous_independent_gaussian',
    'heterogeneous_identical_binary',
    'heterogeneous_independent_gaussian'])

    parser.add_argument("adaptation_mode",
    help='''specify the mode of adaptation: local or global''',
    choices=['local','global'])

    args = parser.parse_args()

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH*0.8,TEXT_WIDTH*0.6))

    plot(ax,args.input_type,args.adaptation_mode)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR, args.input_type
    +'_input_hom_regulation_r_a_'
    +args.adaptation_mode
    +'.pdf'))

    fig.savefig(os.path.join(PLOT_DIR, args.input_type
    +'_input_hom_regulation_r_a_'
    +args.adaptation_mode
    +'.png'),dpi=1000)

    plt.show()
