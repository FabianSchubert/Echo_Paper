import numpy as np

from src.rnn import RNN

from src.testfuncs import gen_in_out_one_in_subs

from tqdm import tqdm

from stdParams import *
import os

from datetime import datetime

import sys

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

parser.add_argument("--N",
help="number of neurons",
type=int,
default=500)

parser.add_argument("--n_sweep_sigm_e",
help="number of sweep steps for external input variance",
type=int,
default=30)

parser.add_argument("--n_sweep_R_t",
help="number of sweep steps for target spectral Radius",
type=int,
default=30)

parser.add_argument("--T_run_adapt",
help="number of time steps for adaptation",
type=int,
default=200000)

parser.add_argument("--T_prerun",
help="number of prerun time steps for learning",
type=int,
default=100)

parser.add_argument("--T_run_learn",
help="number of time steps for learning",
type=int,
default=5000)

parser.add_argument("--T_run_test",
help="number of time steps for testing performance",
type=int,
default=5000)

parser.add_argument("--T_run_ESP",
help="number of time steps for testing the echo state property",
type=int,
default=5000)

parser.add_argument("--T_sample_variance",
help="number of time steps for the sample run",
type=int,
default=5000)

parser.add_argument("--d_init_ESP",
help="initial distance for testing the echo state property",
type=float,
default=1e-3)

parser.add_argument("--tau_max",
help="maximal delay time steps for testing performance",
type=int,
default=15)

parser.add_argument("--y_mean_target",
help="target activity",
type=float,
default=0.05)

parser.add_argument("adapt_mode",
help="set mode for homeostatic adaptation (local or global)",
choices=["local","global"])

args = parser.parse_args()

input_type = ['homogeneous_identical_binary',
'homogeneous_independent_gaussian',
'heterogeneous_identical_binary',
'heterogeneous_independent_gaussian'].index(args.input_type)

adapt_mode = args.adapt_mode

N = args.N
n_sweep_sigm_e = args.n_sweep_sigm_e
n_sweep_R_t = args.n_sweep_R_t
T_run_adapt = args.T_run_adapt
T_prerun = args.T_prerun
T_run_learn = args.T_run_learn
T_run_test = args.T_run_test
T_run_ESP = args.T_run_ESP
T_sample_variance = args.T_sample_variance
d_init_ESP = args.d_init_ESP
tau_max = args.tau_max
y_mean_target = args.y_mean_target

sigm_e = np.linspace(0.,1.5,n_sweep_sigm_e)

R_t = np.linspace(0.,2.0,n_sweep_R_t)

MC = np.ndarray((n_sweep_sigm_e,n_sweep_R_t,tau_max))

W = np.ndarray((n_sweep_sigm_e,n_sweep_R_t,N,N))
a = np.ndarray((n_sweep_sigm_e,n_sweep_R_t,N))
b = np.ndarray((n_sweep_sigm_e,n_sweep_R_t,N))

sigm_x_r_adapt = np.ndarray((n_sweep_sigm_e,n_sweep_R_t,N))
sigm_x_r_test = np.ndarray((n_sweep_sigm_e,n_sweep_R_t,N))

sigm_x_e_adapt = np.ndarray((n_sweep_sigm_e,n_sweep_R_t,N))
sigm_x_e_test = np.ndarray((n_sweep_sigm_e,n_sweep_R_t,N))

for k in tqdm(range(n_sweep_sigm_e)):
    for l in tqdm(range(n_sweep_R_t)):

        rnn = RNN(N=N,y_mean_target=y_mean_target,R_target=R_t[l])

        if input_type == 0:

            rnn.w_in = np.ones((rnn.N,1))

            u_in_adapt,u_out = gen_in_out_one_in_subs(T_run_adapt,1)
            u_in_adapt *= sigm_e[k]
            
            adapt = rnn.run_var_adapt_R(u_in=u_in_adapt,T_skip_rec=1000,adapt_mode=adapt_mode,
                show_progress=False)

            #run test sample
            u_in_adapt_sample,u_out_adapt_sample = gen_in_out_one_in_subs(T_sample_variance,0)
            u_in_adapt_sample *= sigm_e[k]

            y, X_r, X_e = rnn.run_sample(u_in=u_in_adapt_sample,show_progress=False)


        elif input_type == 1:

            rnn.w_in = np.ones((rnn.N,1))
            
            adapt = rnn.run_var_adapt_R(u_in=None,
                                    sigm_e=sigm_e[k],T=T_run_adapt,T_skip_rec=1000,
                                    adapt_mode=adapt_mode,show_progress=False)

            #run sample after adaptation, USING THE INPUT STATISTICS OF THE ADAPTATION!!
            y, X_r, X_e = rnn.run_sample(u_in=None,sigm_e=sigm_e[k],T=T_sample_variance,show_progress=False)
            
            
        elif input_type == 2:

            rnn.w_in = np.random.normal(0.,1.,(N,1))

            u_in_adapt,u_out = gen_in_out_one_in_subs(T_run_adapt,1)
            u_in_adapt *= sigm_e[k]

            adapt = rnn.run_var_adapt_R(u_in=u_in_adapt,T_skip_rec=1000,adapt_mode=adapt_mode,
                show_progress=False)

            #run test sample
            u_in_adapt_sample,u_out_adapt_sample = gen_in_out_one_in_subs(T_sample_variance,0)
            u_in_adapt_sample *= sigm_e[k]

            y, X_r, X_e = rnn.run_sample(u_in=u_in_adapt_sample,show_progress=False)

        else:

            rnn.w_in = np.random.normal(0.,1.,(N,1))

            sigm_e_dist = np.abs(rnn.w_in[:,0]) * sigm_e[k]

            adapt = rnn.run_var_adapt_R(u_in=None,sigm_e=sigm_e_dist,T=T_run_adapt,T_skip_rec=1000,
                adapt_mode=adapt_mode,show_progress=False)

            #run sample after adaptation, USING THE INPUT STATISTICS OF THE ADAPTATION!!
            y, X_r, X_e = rnn.run_sample(u_in=None,sigm_e=sigm_e_dist,T=T_sample_variance,show_progress=False)

        sigm_x_r_adapt[k,l,:] = X_r.std(axis=0)
        sigm_x_e_adapt[k,l,:] = X_e.std(axis=0)


        W[k,l,:,:] = rnn.W
        a[k,l,:] = rnn.a_r
        b[k,l,:] = rnn.b

        for tau in range(tau_max):

            u_in_learn,u_out_learn = gen_in_out_one_in_subs(T_run_learn+T_prerun,tau)
            u_in_learn *= sigm_e[k]

            rnn.learn_w_out_trial(u_in_learn,u_out_learn,reg_fact=.01,show_progress=False,T_prerun=T_prerun)

            u_in_test,u_out_test = gen_in_out_one_in_subs(T_run_test+T_prerun,tau)
            u_in_test *= sigm_e[k]

            u_out_pred = rnn.predict_data(u_in_test,show_progress=False)

            MC[k,l,tau] = np.corrcoef(u_out_test[T_prerun:],u_out_pred[T_prerun:])[0,1]**2.


        #run test sample
        u_in_test,u_out_test = gen_in_out_one_in_subs(T_sample_variance,0)
        u_in_test *= sigm_e[k]
        y, X_r, X_e = rnn.run_sample(u_in=u_in_test,show_progress=False)

        sigm_x_r_test[k,l,:] = X_r.std(axis=0)
        sigm_x_e_test[k,l,:] = X_e.std(axis=0)

if not(os.path.isdir(os.path.join(DATA_DIR, args.input_type+'_input_ESN/performance_sweep/'))):
    os.makedirs(os.path.join(DATA_DIR, args.input_type+'_input_ESN/performance_sweep/'))

np.savez(os.path.join(DATA_DIR, args.input_type+'_input_ESN/performance_sweep/param_sweep_performance_R_t_'
        +adapt_mode + "_"
        +str(datetime.now().isoformat())+'.npz'),
        R_t=R_t,
        sigm_e=sigm_e,
        sigm_x_r_adapt=sigm_x_r_adapt,
        sigm_x_r_test=sigm_x_r_test,
        sigm_x_e_adapt=sigm_x_e_adapt,
        sigm_x_e_test=sigm_x_e_test,
        W=W,
        a=a,
        b=b,
        MC=MC)
