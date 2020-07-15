import numpy as np

from src.rnn import RNN

from src.testfuncs import gen_in_out_one_in_subs

from scipy.sparse import csr_matrix

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
default=None)

parser.add_argument("--n_sweep_R_a",
help="number of sweep steps for spectral Radius",
type=int,
default=30)

parser.add_argument("--T_run_adapt",
help="number of time steps for adaptation",
type=int,
default=1000)

parser.add_argument("--T_sample",
help="number of time steps for the sample run",
type=int,
default=1000)

parser.add_argument("--T_prerun",
help="number of prerun time steps for sample run",
type=int,
default=100)

parser.add_argument("--y_mean_target",
help="target activity",
type=float,
default=0.05)

parser.add_argument("--n_samples",
help="Number of sample runs",
type=int,
default=10)

args = parser.parse_args()

input_type = args.input_type

N = args.N

n_sweep_sigm_e = args.n_sweep_sigm_e
if (n_sweep_sigm_e == None):
   sigm_e = np.array([0.,0.5,1.5])
   n_sweep_sigm_e = 3
else:
   sigm_e = np.linspace(0.,1.5,n_sweep_sigm_e)
   
n_sweep_R_a = args.n_sweep_R_a
T_run_adapt = args.T_run_adapt
T_sample = args.T_sample
T_prerun = args.T_prerun
n_samples = args.n_samples
y_mean_target = args.y_mean_target

R_a = np.linspace(0.,2.0,n_sweep_R_a)

Corr_av_samples = np.ndarray((n_sweep_sigm_e,n_sweep_R_a,n_samples))
    
for k in tqdm(range(n_sweep_sigm_e)):

   for l in tqdm(range(n_sweep_R_a)):

      for m in tqdm(range(n_samples),disable=True):
         
         rnn = RNN(N=N,y_mean_target=y_mean_target,eps_a_r=0.)
         
         rnn.W *= R_a[l] / np.abs(np.linalg.eigvals(rnn.W)).max()
         
         if(input_type == "heterogeneous_identical_binary"):
            
            w_in = np.random.normal(0.,1.,(N,1))
            rnn.w_in = w_in

            u_in_adapt,u_out = gen_in_out_one_in_subs(T_run_adapt,1)
            u_in_adapt *= sigm_e[k]
           
            adapt = rnn.run_hom_adapt(u_in=u_in_adapt,T_skip_rec=1000,show_progress=False)
            
            u_in_sample,u_out = gen_in_out_one_in_subs(T_sample,1)
            u_in_sample *= sigm_e[k]
            
            sample = rnn.run_sample(u_in=u_in_sample,show_progress=False)
         
         if(input_type == "heterogeneous_independent_gaussian"):
            
            w_in = np.random.normal(0.,1.,(N,1))
            sigm_e_dist = np.abs(w_in[:,0]) * sigm_e[k]
            rnn.w_in = w_in
           
            adapt = rnn.run_hom_adapt(sigm_e = sigm_e_dist,T=T_run_adapt,T_skip_rec=1000,show_progress=False)
            
            sample = rnn.run_sample(sigm_e = sigm_e_dist,T=T_sample,show_progress=False)
         
         if(input_type == "homogeneous_identical_binary"):
            
            w_in = np.ones((N,1))
            rnn.w_in = w_in

            u_in_adapt,u_out = gen_in_out_one_in_subs(T_run_adapt,1)
            u_in_adapt *= sigm_e[k]
           
            adapt = rnn.run_hom_adapt(u_in=u_in_adapt,T_skip_rec=1000,show_progress=False)
            
            u_in_sample,u_out = gen_in_out_one_in_subs(T_sample,1)
            u_in_sample *= sigm_e[k]
            
            sample = rnn.run_sample(u_in=u_in_sample,show_progress=False)
         
         if(input_type == "homogeneous_independent_gaussian"):
            
            w_in = np.ones((N,1))
            sigm_e_dist = np.abs(w_in[:,0]) * sigm_e[k]
            rnn.w_in = w_in
           
            adapt = rnn.run_hom_adapt(sigm_e = sigm_e_dist,T=T_run_adapt,T_skip_rec=1000,show_progress=False)
            
            sample = rnn.run_sample(sigm_e = sigm_e_dist,T=T_sample,show_progress=False)
            
         y = sample[0]
         
         C = np.corrcoef(y[T_prerun:].T)
                 
         C_av = (np.abs(C).sum() - np.abs(C[range(N),range(N)]).sum())/(N**2. - N)

         Corr_av_samples[k,l,m] = C_av  
         

if not(os.path.isdir(os.path.join(DATA_DIR, args.input_type+'_input_ESN'))):
    os.makedirs(os.path.join(DATA_DIR, args.input_type+'_input_ESN'))

np.savez(os.path.join(DATA_DIR, args.input_type + '_input_ESN/corr_R_a.npz'),
   R_sweep = R_a,
   sigm_ext_sweep=sigm_e,
   Corr_av_samples = Corr_av_samples)