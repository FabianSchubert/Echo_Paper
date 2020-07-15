import numpy as np
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
default=1.)

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
T_rec = int(T/T_skip_rec)

t_arr = np.arange(T_rec)*T_skip_rec


### recording
a_rec = np.ones((n_samples,T_rec,N))
b_rec = np.ones((T_rec,N))

y_rec = np.ones((T_rec,N))
y_norm_rec = np.ones((n_samples,T_rec))
X_r_rec = np.ones((T_rec,N))
X_e_rec = np.ones((T_rec,N))

y_squ_trail_av_rec = np.ones((T_rec,N))

mu_y_rec = np.ones((T_rec,N))
mu_X_e_rec = np.ones((T_rec,N))
Var_y_rec = np.ones((T_rec,N))
Var_X_e_rec = np.ones((T_rec,N))
###

for k in tqdm(range(n_samples)):

    W = np.random.normal(0.,1./(cf_w*N)**.5,(N,N)) * (np.random.rand(N,N) <= cf_w)
    W[range(N),range(N)] = 0.

    W = W/np.max(np.abs(np.linalg.eigvals(W)))

    W_av = 1.*(W!=0.)
    W_av = (W_av.T / W_av.sum(axis=1)).T

    if rand_a_init:
        a = np.ones((N)) * np.random.rand() * a_init
    else:
        a = np.ones((N))*a_init

    b = np.zeros((N))

    if sigm_w_e > 0.:
        if (input_type in [0,1]):
            w_in = np.ones(N) * (np.random.rand(N) <= cf_w_in) * (sigm_w_e/cf_w_in**.5)
        else:
            w_in = np.random.normal(0.,1.,(N)) * (np.random.rand(N) <= cf_w_in) * (sigm_w_e/cf_w_in**.5)
    else:
        w_in = np.zeros((N))

    y = np.ndarray((N))
    X_r = np.ndarray((N))
    X_e = np.ndarray((N))

    y_squ_trail_av = np.ndarray((N))

    ##trailing average of X_r**2 for adjusting the learning rate
    X_r_squ_av = X_r**2.

    #X_e = (w_in @ u_in).T
    #X_e = np.random.normal(0.,1.,(T,N)) * w_in[:,0]
    #X_e = np.random.normal(0.,.25,(T,N))

    mu_y = np.ndarray((N))
    mu_X_e = np.ndarray((N))
    Var_y = np.ndarray((N))
    Var_X_e = np.ndarray((N))



    ### first time step
    if input_type in [0,2]:
        X_e[:] = w_in * (2.*(np.random.rand() <= 0.5) - 1.)
    else:
        X_e[:] = w_in * np.random.normal(0.,1.,(N))
    X_r[:] = (np.random.rand(N)-.5)
    X_r[:] *= np.random.rand()*X_r_norm_init_span/np.linalg.norm(X_r)
    y[:] = np.tanh(X_r[:] + X_e[:])
    y_squ_trail_av[:] = y**2.

    mu_y[:] = y
    mu_X_e[:] = X_e
    Var_y[:] = .25
    Var_X_e[:] = .25

    #### Recording
    a_rec[k,0,:] = a
    b_rec[0,:] = b

    y_rec[0,:] = y
    y_norm_rec[k,0] = np.linalg.norm(y)
    X_r_rec[0,:] = X_r
    X_e_rec[0,:] = X_e

    y_squ_trail_av_rec[0,:] = y_squ_trail_av

    mu_y_rec[0,:] = mu_y
    mu_X_e_rec[0,:] = mu_X_e
    Var_y_rec[0,:] = Var_y
    Var_X_e_rec[0,:] = Var_X_e
    ####
    ###

    for t in tqdm(range(1,T)):

        y_prev = y[:]

        X_r[:] = a[:] * (W @ y[:])

        if input_type in [0,2]:
            X_e[:] = w_in * (2.*(np.random.rand() <= 0.5) - 1.)
        else:
            X_e[:] = w_in * np.random.normal(0.,1.,(N))

        y[:] = np.tanh(X_r + X_e - b)

        mu_y[:] = (1.-eps_mu)*mu_y + eps_mu * y
        mu_X_e[:] = (1.-eps_mu)*mu_X_e + eps_mu * X_e

        Var_y[:] = (1.-eps_var)*Var_y + eps_var * (y - mu_y)**2.
        Var_X_e[:] = (1.-eps_var)*Var_X_e + eps_var * (X_e - mu_X_e)**2.

        y_squ_trail_av = (1.-eps_y_squ)*y_squ_trail_av + eps_y_squ * y_prev**2.

        X_r_squ_av += eps_X_r_squ*(X_r**2.- X_r_squ_av)

        #y_squ_targ = 1.-1./(1.+2.*Var_y.mean() + 2.*Var_X_e)**.5

        #a = a + eps_a * a * ((y**2.).mean() - (X_r**2.).mean())
        if adaptation_mode == 0:
            a = a + eps_a * a * (y_squ_trail_av - X_r_squ_av)/X_r_squ_av.mean()
        else:
            a = a + eps_a * a * (y_squ_trail_av.mean() - X_r_squ_av.mean())/X_r_squ_av.mean()
        #a = a + eps_a * (W_av @ (y_prev**2.) - X_r**2.)
        #a = a + eps_a * ((y**2.) - (X_r**2.))
        b = b + eps_b * (y - mu_y_target)

        a = np.maximum(0.001,a)

        if t%T_skip_rec == 0:
            t_rec = int(t/T_skip_rec)

            #### Recording
            a_rec[k,t_rec,:] = a
            b_rec[t_rec,:] = b

            y_rec[t_rec,:] = y
            X_r_rec[t_rec,:] = X_r
            X_e_rec[t_rec,:] = X_e

            mu_y_rec[t_rec,:] = mu_y
            mu_X_e_rec[t_rec,:] = mu_X_e
            Var_y_rec[t_rec,:] = Var_y
            Var_X_e_rec[t_rec,:] = Var_X_e

            y_squ_trail_av_rec[t_rec,:] = y_squ_trail_av
            ####
    y_norm_rec[k,:] = np.linalg.norm(y_rec,axis=1)

DATA_DIR = '/home/fabian/work/data/'

if not(os.path.isdir(os.path.join(DATA_DIR, args.input_type + '_input_ESN/alt_hom_regulation/'))):
    os.makedirs(os.path.join(DATA_DIR, args.input_type + '_input_ESN/alt_hom_regulation/'))

np.savez(os.path.join(DATA_DIR,  args.input_type
        +'_input_ESN/alt_hom_regulation/alt_hom_regulation_'
        +args.adaptation_mode
        +'_'+str(datetime.now().isoformat())+'.npz'),
        a=a_rec,
        b=b_rec,
        W=W,
        y_norm=y_norm_rec,
        y=y_rec,
        N=N,
        n_samples=n_samples,
        cf_w = cf_w,
        cf_w_in = cf_w_in,
        sigm_w_e =sigm_w_e,
        eps_a = eps_a,
        eps_b = eps_b,
        mu_y_target = mu_y_target,
        X_r=X_r_rec,
        y_squ_trail_av=y_squ_trail_av_rec)
