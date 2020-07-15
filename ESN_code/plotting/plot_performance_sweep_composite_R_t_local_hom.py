#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

plt.rc('text.latex', preamble=r'''
\usepackage{dejavu}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}''')

from stdParams import *
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hide_plot",action="store_true")

args = parser.parse_args()

import ESN_code.plotting.plot_performance_sweep_R_t as plot_performance_sweep

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.45))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

print("plotting performance sweep homogeneous_independent_gaussian...")
plot_performance_sweep.plot(ax2,'homogeneous_independent_gaussian','local')
print("plotting performance sweep homogeneous_identical_binary...")
plot_performance_sweep.plot(ax1,'homogeneous_identical_binary','local')

for k in range(2):

   fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

   ax2_title = '\\makebox['+str(ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont homogeneous gauss}'
   ax1_title = '\\makebox['+str(ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont homogeneous binary}'

   ax1.set_title(ax1_title,loc='left',usetex=True)
   ax2.set_title(ax2_title,loc='left',usetex=True)

   fig.tight_layout(pad=0.1,h_pad=0.5,w_pad=0.5)

fig.savefig(os.path.join(PLOT_DIR,'performance_sweep_composite_R_t_local_hom.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'performance_sweep_composite_R_t_local_hom.png'),dpi=1000)

#fig.savefig(os.path.join(PLOT_DIR,'r_a_sweep_composite_low_res.png'),dpi=300)
if not(args.hide_plot):
    plt.show()
