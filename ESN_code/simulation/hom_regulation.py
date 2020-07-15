import numpy as np
from tqdm import tqdm

from stdParams import *
import os

from datetime import datetime

from src.rnn import RNN 

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

parser.add_argument("adaptation_mode",
help='''specify the mode of adaptation: local or global''',
choices=['local','global'])

parser.add_argument("--N",
help='''number of Neurons''',
type=int,
default=500)

parser.add_argument("--n_samples",
help='''number of sample runs''',
type=int,
default=1)

parser.add_argument("--cf_w",
help='''recurrent connection fraction''',
type=float,
default=.1)

parser.add_argument("--cf_w_in",
help='''external input connection fraction''',
type=float,
default=1.)

parser.add_argument("--sigm_w_e",
help='''external input standard deviation''',
type=float,
default=.5)

parser.add_argument("--eps_a",
help='''gain learning rate''',
type=float,
default=1e-3)

parser.add_argument("--eps_b",
help='''bias learning rate''',
type=float,
default=1e-4)

parser.add_argument("--eps_mu",
help='''activity trailing average adaptation rate''',
type=float,
default=1e-3)

parser.add_argument("--eps_var",
help='''activity variance trailing average adaptation rate''',
type=float,
default=1e-2)

parser.add_argument("--eps_y_squ",
help='''squared local activity trailing averate adaptation rate''',
type=float,
default=1e-2)

parser.add_argument("--eps_X_r_squ",
help='''adaptation rate for trailing average of squared recurrent membrane potential''',
type=float,
default=1e-2)

parser.add_argument("--mu_y_target",
help='''activity target''',
type=float,
default=0.05)

parser.add_argument("--a_init",
help='''initial gain''',
type=float,
default=1.5)

parser.add_argument("--T",
help='''runtime''',
type=int,
default=int(3e4))

parser.add_argument("--T_skip_rec",
help='''time sampling skip length for recording (1 means record everything)''',
type=int,
default=1)

parser.add_argument("--X_r_norm_init_span",
help='''Maximum norm of randomized initial X_r''',
type=float,
default=100.)

parser.add_argument("--rand_a_init",
help='''if true, initial gain is randomized between 0 and a_init''',
type=bool,
default=False)

args = parser.parse_args()

input_type = ['homogeneous_identical_binary',
'homogeneous_independent_gaussian',
'heterogeneous_identical_binary',
'heterogeneous_independent_gaussian'].index(args.input_type)

adaptation_mode = ['local','global'].index(args.adaptation_mode)

N = args.N

n_samples = args.n_samples

cf_w = args.cf_w
cf_w_in = args.cf_w_in

sigm_w_e = args.sigm_w_e

eps_a = args.eps_a
eps_b = args.eps_b

eps_mu = args.eps_mu
eps_var = args.eps_var
eps_y_squ = args.eps_y_squ
eps_X_r_squ = args.eps_X_r_squ

mu_y_target = np.ones((N))*args.mu_y_target

a_init = args.a_init
X_r_norm_init_span = args.X_r_norm_init_span

rand_a_init = args.rand_a_init

#r_target = .9

T = args.T
T_skip_rec = args.T_skip_rec

### recording
a_rec = []
b_rec = []

y_rec = []
X_r_rec = []
X_e_rec = []

W_rec = []
###



for k in tqdm(range(n_samples)):
    
    if rand_a_init:
        a = np.ones((N)) * np.random.rand() * a_init
    else:
        a = np.ones((N))*a_init
    
    if sigm_w_e > 0.:
        if (input_type in [0,1]):
            w_in = np.ones((N,1)) * (np.random.rand(N,1) <= cf_w_in) * (1./cf_w_in**.5)
        else:
            w_in = np.random.normal(0.,1.,(N,1)) * (np.random.rand(N,1) <= cf_w_in) * (1./cf_w_in**.5)
    else:
        w_in = np.zeros((N,1))    
    
    rnn = RNN(N=N,
        cf=cf_w,
        a_r = a,
        eps_a_r = eps_a,
        eps_b = eps_b,
        eps_y_mean = eps_mu,
        eps_y_std = eps_var,
        eps_E_mean = eps_mu,
        eps_E_std = eps_var,
        y_mean_target = mu_y_target,
        R_target = 1.)
    
    rnn.a_r = a
    rnn.w_in = w_in
    
    if input_type in [0,2]:
        u_in = (2.*(np.random.rand(T) <= 0.5) - 1.) * sigm_w_e
        
        y_temp, X_r_temp, X_e_temp, a_r_temp, b_temp, y_mean_temp, y_std_temp = rnn.run_var_adapt_R(u_in=u_in,T_skip_rec = T_skip_rec,adapt_mode=args.adaptation_mode)
        
        
    else:
        if (input_type == 1):
            y_temp, X_r_temp, X_e_temp, a_r_temp, b_temp, y_mean_temp, y_std_temp = rnn.run_var_adapt_R(sigm_e=sigm_w_e,T=T,T_skip_rec = T_skip_rec,adapt_mode=args.adaptation_mode)
        else:
            y_temp, X_r_temp, X_e_temp, a_r_temp, b_temp, y_mean_temp, y_std_temp = rnn.run_var_adapt_R(sigm_e=np.abs(np.random.normal(0.,sigm_w_e,(N))),T=T,T_skip_rec = T_skip_rec,adapt_mode=args.adaptation_mode)
    
    
    a_rec.append(a_r_temp)
    b_rec.append(b_temp)

    y_rec.append(y_temp)
    X_r_rec.append(X_r_temp)
    X_e_rec.append(X_e_temp)
    W_rec.append(rnn.W)

a_rec = np.array(a_rec)
b_rec = np.array(b_rec)
y_rec = np.array(y_rec)
X_r_rec = np.array(X_r_rec)
W_rec = np.array(W_rec)


if not(os.path.isdir(os.path.join(DATA_DIR, args.input_type + '_input_ESN/hom_regulation/'))):
    os.makedirs(os.path.join(DATA_DIR, args.input_type + '_input_ESN/hom_regulation/'))

np.savez(os.path.join(DATA_DIR,  args.input_type
        +'_input_ESN/hom_regulation/hom_regulation_'
        +args.adaptation_mode
        +'_'+str(datetime.now().isoformat())+'.npz'),
        a=a_rec,
        b=b_rec,
        W=W_rec,
        y=y_rec,
        N=N,
        n_samples=n_samples,
        cf_w = cf_w,
        cf_w_in = cf_w_in,
        sigm_w_e =sigm_w_e,
        eps_a = eps_a,
        eps_b = eps_b,
        mu_y_target = mu_y_target,
        X_r=X_r_rec)

    
