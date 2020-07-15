#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')
ModCols = mpl.cm.get_cmap('viridis',512)(np.linspace(0.,.7,512))
ModCm = mpl.colors.ListedColormap(ModCols)

from stdParams import *
import os
import glob
from pathlib import Path

from src.analysis_tools import get_simfile_prop

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot(ax,input_type,col=colors[0]):

    #check if there is already a saved dataframe...

    MAE_df = pd.read_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/var_predict_scaling_fix_R_a_df.h5'), 'table')
    '''
    try:
        MSE_df = pd.read_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/var_predict_scaling_df.h5'), 'table')
    except:

        print("No dataframe found! Creating it...")

        #####
        sizes = [100,200,300,400,500,1000]

        filelist = []

        for size in sizes:
            filelist.append(glob.glob(os.path.join(DATA_DIR, input_type + '_input_ESN/N_'+str(size) + '/param_sweep_*')))
            for k,file in enumerate(filelist[-1]):
                filelist[-1][k] = Path(file).relative_to(DATA_DIR)
        #####

        MSE_df = pd.DataFrame(columns=('sigm_e','sigm_t','N','MSE'))

        for file_search in filelist:

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



            for dat_inst in dat:

                sigm_e = dat_inst['sigm_e']
                sigm_t = dat_inst['sigm_t']
                y = dat_inst['y']
                X_r = dat_inst['X_r']
                W = dat_inst['W']

                sigm_w = W.std(axis=3)
                sigm_X_r = X_r.std(axis=2)
                sigm_y = y.std(axis=2)

                n_sigm_t = sigm_t.shape[0]
                n_sigm_e = sigm_e.shape[0]

                N = y.shape[3]

                for k in range(n_sigm_e):
                    for l in range(n_sigm_t):
                        MSE = np.abs(sigm_X_r[k,l,:]**2.-sigm_y[k,l,:]**2.).mean()

                        MSE_df = MSE_df.append(pd.DataFrame(columns=('sigm_e','sigm_t','N','MSE','MSE_N'),data=np.array([[sigm_e[k],sigm_t[l],N,MSE,MSE*N]])))

        MSE_df.to_hdf(os.path.join(DATA_DIR, input_type + '_input_ESN/var_predict_scaling_df.h5'),'table')


    MSE_df = MSE_df.loc[MSE_df['sigm_e']==.5]
    #MSE_df["N"] = ["$%s$" % x for x in MSE_df["N"].astype('int')]
    '''
    sns.lineplot(ax=ax,x='N',y='MAE',data=MAE_df,color=col,linewidth=1.5)

    #ax.legend().texts[0].set_text('$\\sigma_{\\rm e}$')

    ax.set_xlabel('$N$')
    ax.set_ylabel('$\\left||\\sigma^2_{\\rm bare}-\sigma^2_{\\rm w} \\sigma^2_{\\rm y} \\right||$')

    #ax.set_ylim(bottom=0.)

if __name__ == '__main__':

    fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.6))

    plot(ax,'heterogeneous_identical_binary')

    fig.tight_layout(pad=0.1)

    fig.savefig(os.path.join(PLOT_DIR, 'heterogeneous_identical_binary' + '_input_rec_mem_pot_predict_size_scaling_new.pdf'))
    fig.savefig(os.path.join(PLOT_DIR, 'heterogeneous_identical_binary' + '_input_rec_mem_pot_predict_size_scaling_new.png'),dpi=1000)

    fig.savefig(os.path.join(PLOT_DIR, 'heterogeneous_identical_binary' + '_input_rec_mem_pot_predict_size_scaling_new_low_res.png'),dpi=300)

    plt.show()
