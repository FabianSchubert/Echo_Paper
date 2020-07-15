#!/usr/bin/env python3
import numpy as np
from stdParams import *
import os
import sys
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("input_type",
help='''specify four type of input (homogeneous_identical_binary,
homogeneous_independent_gaussian, heterogeneous_identical_binary,
heterogeneous_independent_gaussian)''',
choices=['homogeneous_identical_binary',
'homogeneous_independent_gaussian',
'heterogeneous_identical_binary',
'heterogeneous_independent_gaussian'])

parser.add_argument("--hide_plot",action='store_true')

def plot(ax,input_type):

    try:
        dat = np.load(os.path.join(DATA_DIR, input_type + '_input_ESN/alt_hom_regulation/flow_data.npz'))
    except:
        print("Could not load data file!")
        sys.exit()

    a_rec = dat['a']
    y_norm_rec = dat['y_norm']
    N=dat['N']
    n_samples=dat['n_samples']
    cf_w = dat['cf_w']
    cf_w_in = dat['cf_w_in']
    sigm_w_e = dat['sigm_w_e']
    print(sigm_w_e)
    eps_a = dat['eps_a']
    print(eps_a)
    eps_b = dat['eps_b']
    mu_y_target = dat['mu_y_target']

    n_averaging_inp = 1000

    a = np.linspace(0.,2.5,500)
    vy = np.linspace(0.,1.,500)

    A,VY = np.meshgrid(a,vy)

    delta_a = eps_a*A*(1.-A**2.)*VY

    delta_vy = np.zeros((500,500))

    for k in tqdm(range(n_averaging_inp)):

        delta_vy += 1-(1. + 2*A**2.*VY + 2.*np.random.normal(0.,sigm_w_e)**2.)**(-.5)-VY

    delta_vy /= n_averaging_inp


    ax.streamplot(A,VY,delta_a,delta_vy)

    vy_pl = np.linspace(0.,1.,1000)
    a_pl = np.linspace(0.,2.5,1000)



    #ax.plot((((1.-vy_pl/N)**(-2.)/2. - v_e - .5 )/(sigm_w**2.*vy_pl/N))**.5,vy_pl)
    #ax.plot(0.*a + sigm_w**(-1.),vy)

    for k in range(n_samples):
        plt.plot(a_rec[k,:,0],y_norm_rec[k,:]**2./N,c=colors[1],alpha=1.,lw=1)
        #plt.plot(a_rec[k,1:,0])
        #plt.plot(y_norm_rec[k,1:])

    ax.contour(a,vy,delta_a,levels=[0.],colors=[colors[2]],linewidths=[2.],zorder=3)
    ax.contour(a,vy,delta_vy,levels=[0.],colors=[colors[3]],linewidths=[2.],zorder=4)

    ax.set_xlim([a_pl[0]-.1,a_pl[-1]])
    ax.set_ylim([vy_pl[0]-.1,vy_pl[-1]])

    ax.set_xlabel('$a$')
    ax.set_ylabel('$\\sigma_{\\rm y}^2$')

if __name__ == '__main__':

    args = parser.parse_args()

    input_type = args.input_type

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

    plot(ax,input_type)

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR, input_type + '_input_alt_hom_regulation_flow.pdf'))
    fig.savefig(os.path.join(PLOT_DIR, input_type + '_input_alt_hom_regulation_flow.png'),dpi=300)
    
    if not(args.hide_plot):
        plt.show()
