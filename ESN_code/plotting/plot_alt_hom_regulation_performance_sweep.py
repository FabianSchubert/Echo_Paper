#!/usr/bin/env python3
import numpy as np
from stdParams import *
import os
import sys
import glob
from pathlib import Path

from src.analysis_tools import get_simfile_prop

import pandas as pd

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

parser.add_argument("adaptation_mode",
help='''specify the mode of adaptation: local or global''',
choices=['local','global'])

def plot(ax,input_type,adaptation_mode):


    try:
        #sweep_df = pd.read_pickle(file_search_preprocess)
        sweep_df = pd.read_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_alt_hom_regulation_processed_data_'
                                    +adaptation_mode
                                    +'.h5'), 'table')
    except:

        file_search = glob.glob(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_alt_hom_regulation_performance_'
                                            +adaptation_mode
                                            +'_*'))
        if isinstance(file_search,list):
            simfile = []
            timestamp = []
            for file_search_inst in file_search:
                simfile_inst, timestamp_inst = get_simfile_prop(os.path.join(DATA_DIR,file_search_inst))
                simfile.append(simfile_inst)
                timestamp.append(timestamp_inst)
        else:
            simfile,timestamp = get_simfile_prop(os.path.join(DATA_DIR,file_search))
            simfile = [simfile]
            timestamp = [timestamp]

        dat = []

        for simfile_inst in simfile:
            dat.append(np.load(simfile_inst))

        sweep_df = pd.DataFrame(columns=('sigm_e','r_a','MC_abs','specrad','timestamp'))

        for i,dat_inst in enumerate(dat):

            sigm_e = dat_inst['sigm_e']
            r_a = dat_inst['r_a']

            a = dat_inst['a']
            W = dat_inst['W']

            MC = dat_inst['MC']

            n_samples = MC.shape[0]
            n_r_a = r_a.shape[0]
            n_sigm_e = sigm_e.shape[0]





            print('Processing data...')
            for n in tqdm(range(n_samples)):
                for k in tqdm(range(n_sigm_e)):
                    for l in range(n_r_a):

                        MC_abs = MC[n].sum(axis=2)

                        specrad = np.abs(np.linalg.eigvals((a[n,k,l,:] * W[n,k,l,:,:].T).T)).max()

                        sweep_df = sweep_df.append(pd.DataFrame(columns=('sigm_e','r_a','MC_abs','specrad','timestamp'),data=np.array([[sigm_e[k],r_a[l],MC_abs[k,l],specrad,timestamp[i]]])))

        sweep_df.sigm_e = sweep_df.sigm_e.astype('float')
        sweep_df.r_a = sweep_df.r_a.astype('float')
        sweep_df.MC_abs = sweep_df.MC_abs.astype('float')
        sweep_df.specrad = sweep_df.specrad.astype('float')
        sweep_df.timestamp = sweep_df.timestamp.astype('datetime64')

        sweep_df = sweep_df.reset_index()

        sweep_df.to_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/performance_sweep/param_sweep_performance_alt_hom_regulation_processed_data_'
                                    +adaptation_mode
                                    +'.h5'),'table')

    sigm_e = np.array(sweep_df.sigm_e.unique())
    r_a = np.array(sweep_df.r_a.unique())

    sigm_e_ax = np.append(sigm_e,2.*sigm_e[-1] - sigm_e[-2]) - 0.5*(sigm_e[1] - sigm_e[0])
    r_a_ax = np.append(r_a,2.*r_a[-1] - r_a[-2]) - 0.5*(r_a[1] - r_a[0])

    sweep_df_group = sweep_df.groupby(by=['sigm_e','r_a'])

    sweep_df_mean = sweep_df_group.mean()
    sweep_df_mean.reset_index(inplace=True)

    sweep_df_sem = sweep_df_group.agg('sem')
    sweep_df_sem.reset_index(inplace=True)

    sweep_df_merge = pd.merge(sweep_df_mean,sweep_df_sem,on=['sigm_e','r_a'],suffixes=['_mean','_sem'])

    sweep_df_group_sigm_e_timestamp = sweep_df.groupby(by=['sigm_e','timestamp'])

    max_MC_idx = sweep_df_group_sigm_e_timestamp.idxmax()

    max_MC_values = sweep_df.loc[max_MC_idx.MC_abs]

    max_MC_values_mean = max_MC_values.groupby(by=['sigm_e']).mean()
    max_MC_values_sem = max_MC_values.groupby(by=['sigm_e']).agg('sem')

    #sweep_df_max_sigm_t = sweep_df_group_sigm_e.agg('max')
    #sweep_df_max_sigm_t.reset_index(inplace=True)

    MC_pivot = sweep_df_merge.pivot(index='sigm_e',columns='r_a',values='MC_abs_mean')

    ### Cutoff for masking is 0.2
    pcm = ax.pcolormesh(r_a_ax,sigm_e_ax,np.ma.MaskedArray(MC_pivot,MC_pivot < 2e-1),cmap='viridis',rasterized=True,vmin=0.,vmax=9.)

    plt.colorbar(ax=ax,mappable=pcm)

    ax.contour(r_a,sigm_e,sweep_df_merge.pivot(index='sigm_e',columns='r_a',values='specrad_mean'),levels=[1.],linestyles=['dashed'],colors=['w'],linewidths=[2.])

    ax.plot(max_MC_values_mean.r_a.to_numpy()[1:],sigm_e[1:],lw=2.,c=BRIGHT_YELLOW)
    ax.fill_betweenx(sigm_e[1:],(max_MC_values_mean.r_a-max_MC_values_sem.r_a).to_numpy()[1:],(max_MC_values_mean.r_a+max_MC_values_sem.r_a).to_numpy()[1:],color=BRIGHT_YELLOW,alpha=.25)

    ax.set_xlabel("$R_{\\rm t}$")
    ax.set_ylabel("$\\sigma_{\\rm ext}$")



if __name__ == '__main__':

    args = parser.parse_args()

    fig, ax = plt.subplots(1,1)

    plot(ax,args.input_type,args.adaptation_mode)

    plt.show()
