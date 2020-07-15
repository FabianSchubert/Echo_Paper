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

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from stdParams import *
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hide_plot",action='store_true')

args = parser.parse_args()


import ESN_code.plotting.plot_alt_hom_regulation_check_conv_cond as check_conv_cond
import ESN_code.plotting.plot_alt_hom_regulation_r_a as r_a
import ESN_code.plotting.plot_alt_hom_regulation_specrad as specrad

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.75))

#fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.8))

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

r_a.plot(ax1,'heterogeneous_independent_gaussian','local',col=colors[0])

r_a.plot(ax2,'heterogeneous_identical_binary','local',col=colors[1])
ax2.set_ylim([0.,5.])

r_a.plot(ax3,'heterogeneous_identical_binary','global',col=colors[2])

specrad.plot(ax4,'heterogeneous_independent_gaussian','local','A',col=colors[0])
specrad.plot(ax4,'heterogeneous_identical_binary','local','B',col=colors[1])
specrad.plot(ax4,'heterogeneous_identical_binary','global','C',col=colors[2])

ax4.set_ylim([-1.8,1.8])

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

ax1_title = '\\makebox['+str(ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont heterogeneous gaussian (local)}'
ax2_title = '\\makebox['+str(ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont heterogeneous binary (local)}'
ax3_title = '\\makebox['+str(ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf C} \\hfill \\normalfont heterogeneous binary (global)}'
ax4_title = '\\makebox['+str(ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf D} \\hfill \\normalfont complex eigenvalues}'

ax1.set_title(ax1_title,loc='left',usetex=True)
ax2.set_title(ax2_title,loc='left',usetex=True)
ax3.set_title(ax3_title,loc='left',usetex=True)
ax4.set_title(ax4_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_composite.pdf'))
fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_composite.png'),dpi=1000)

#fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_composite_low_res.png'),dpi=300)
if not(args.hide_plot):
   plt.show()
