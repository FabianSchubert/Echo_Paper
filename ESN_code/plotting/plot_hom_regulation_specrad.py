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

import matplotlib.patches as mpatches

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from tqdm import tqdm

import argparse

def plot(ax,input_type,adaptation_mode,label_str,col=colors[0]):

    file = get_simfile_prop(os.path.join(DATA_DIR,input_type
    +'_input_ESN/hom_regulation/hom_regulation_'+adaptation_mode))

    dat = np.load(file[0])

    a_rec = dat['a']
    N=dat['N']
    W = dat['W']
    
    #import pdb
    #pdb.set_trace()
    
    l_start = np.linalg.eigvals((W[0].T * a_rec[0,0,:]).T)
    l_end = np.linalg.eigvals((W[0].T * a_rec[0,-1,:]).T)

    #ax.plot(l_start.real,l_start.imag,'.',markersize=5,label='$t=0$')
    #sc_not_fact = a_rec.shape[1]/10**sc_not_exp
    #sc_not_exp = int(np.log10(a_rec.shape[1]))
    #ax.plot(l_end.real,l_end.imag,'.',markersize=5,label='$t='+str(sc_not_fact)+'\\times 10^'+str(sc_not_exp)+'$')
    ax.plot(l_end.real,l_end.imag,'.',markersize=4,alpha=0.8,c=col,label=label_str)
    circle = plt.Circle((0,0),np.abs(l_end).max(),facecolor=(0,0,0,0),edgecolor=col,lw=1.5,linestyle='--')
    ax.add_artist(circle)

    ax.set_xlabel('$\\mathrm{Re}(\\lambda_i)$')
    ax.set_ylabel('$\\mathrm{Im}(\\lambda_i)$')

    ax.legend()

    ax.axis('equal')

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

    plot(ax,args.input_type,args.adaptation_mode,args.input_type+args.adaptation_mode)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR, args.input_type
    +'_input_hom_regulation_specrad_'
    +args.adaptation_mode
    +'.pdf'))

    fig.savefig(os.path.join(PLOT_DIR, args.input_type
    +'_input_hom_regulation_specrad_'
    +args.adaptation_mode
    +'.png'),dpi=1000)

    plt.show()
