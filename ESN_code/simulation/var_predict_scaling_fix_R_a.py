import numpy as np

from src.rnn import RNN

from src.testfuncs import gen_in_out_one_in_subs

from tqdm import tqdm

from stdParams import *
import os

from datetime import datetime

import sys

import pandas as pd

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

parser.add_argument("--T_run_adapt",
help="number of time steps for adaptation",
type=int,
default=10000)

parser.add_argument("--T_prerun_sample",
help="number of prerun time steps before recording a sample",
type=int,
default=100)

parser.add_argument("--T_run_sample",
help="time steps for recording a sample",
type=int,
default=1000)

parser.add_argument("--y_mean_target",
help="target activity",
type=float,
default=0.05)

parser.add_argument("--sigm_e",
help="standard deviation of external driving",
type=float,
default=0.5)

parser.add_argument("--n_samples",
help="number of runs to average over for each data point.",
type=int,
default=10)

args = parser.parse_args()

input_type = ['homogeneous_identical_binary',
'homogeneous_independent_gaussian',
'heterogeneous_identical_binary',
'heterogeneous_independent_gaussian'].index(args.input_type)

N_list = [100,200,300,400,500,700,1000]
n_N = len(N_list)

n_samples = args.n_samples

sigm_e = args.sigm_e
T_run_adapt = args.T_run_adapt
T_prerun_sample = args.T_prerun_sample
T_run_sample = args.T_run_sample
y_mean_target = args.y_mean_target

#####################################

# Mean Absolute Error
MAE_list = []

####################################

for k in tqdm(range(n_N)):

    N = N_list[k]

    for l in tqdm(range(n_samples)):

        rnn = RNN(N=N,y_mean_target=y_mean_target,y_std_target=1.)
        
        rnn.W /= np.abs(np.linalg.eigvals(rnn.W)).max()
        
        rnn.eps_a_r = 0.

        ##################

        if input_type == 0:

            rnn.w_in = np.ones((rnn.N,1))

            u_in_adapt,u_out = gen_in_out_one_in_subs(T_run_adapt,1)
            u_in_adapt *= sigm_e

            adapt = rnn.run_hom_adapt(u_in=u_in_adapt,T_skip_rec=1000,show_progress=False)

            #run test sample
            u_in_sample,u_out_sample = gen_in_out_one_in_subs(T_run_sample+T_prerun_sample,0)
            u_in_sample *= sigm_e

            y_res,X_r_res,X_e_res = rnn.run_sample(u_in=u_in_sample,show_progress=False)


        elif input_type == 1:

            rnn.w_in = np.ones((rnn.N,1))

            adapt = rnn.run_hom_adapt(u_in=None,
                                    sigm_e=sigm_e,T=T_run_adapt,T_skip_rec=1000,show_progress=False)

            #run sample after adaptation, USING THE INPUT STATISTICS OF THE ADAPTATION!!
            y_res,X_r_res,X_e_res = rnn.run_sample(u_in=None,sigm_e=sigm_e,T=T_run_sample+T_prerun_sample,show_progress=False)

        elif input_type == 2:

            rnn.w_in = np.random.normal(0.,1.,(N,1))

            u_in_adapt,u_out = gen_in_out_one_in_subs(T_run_adapt,1)
            u_in_adapt *= sigm_e

            adapt = rnn.run_hom_adapt(u_in=u_in_adapt,T_skip_rec=1000,show_progress=False)

            #run test sample
            u_in_sample,u_out_sample = gen_in_out_one_in_subs(T_run_sample+T_prerun_sample,0)
            u_in_sample *= sigm_e

            y_res,X_r_res,X_e_res = rnn.run_sample(u_in=u_in_sample,show_progress=False)

        else:

            rnn.w_in = np.random.normal(0.,1.,(N,1))

            sigm_e_dist = np.abs(rnn.w_in[:,0]) * sigm_e

            adapt = rnn.run_hom_adapt(u_in=None,sigm_e=sigm_e_dist,T=T_run_adapt,T_skip_rec=1000,show_progress=False)

            #run sample after adaptation, USING THE INPUT STATISTICS OF THE ADAPTATION!!
            y_res,X_r_res,X_e_res = rnn.run_sample(u_in=None,sigm_e=sigm_e_dist,T=T_run_sample+T_prerun_sample,show_progress=False)

        ####################################
        '''
        y_list.append(y_res[T_prerun_sample:,:])
        X_r_list.append(X_r_res[T_prerun_sample:,:])
        X_e_list.append(X_e_res[T_prerun_sample:,:])

        W_list.append(rnn.W)
        a_list.append(rnn.a_r)
        b_list.append(rnn.b)
        '''

        Var_X_r = X_r_res[T_prerun_sample:,:].var(axis=0)
        Var_y = y_res[T_prerun_sample:,:].var()
        Var_W = rnn.W.var(axis=1) * rnn.N
        
        MAE = np.abs(Var_X_r - Var_y*Var_W).mean()

        MAE_list.append({'N':N,'MAE':MAE})

MAE_df = pd.DataFrame(MAE_list, columns=('N','MAE'))

################################

if not(os.path.isdir(os.path.join(DATA_DIR, args.input_type+'_input_ESN'))):
    os.makedirs(os.path.join(DATA_DIR, args.input_type+'_input_ESN'))

MAE_df.to_hdf(os.path.join(DATA_DIR, args.input_type + '_input_ESN/var_predict_scaling_fix_R_a_df.h5'),'table')
