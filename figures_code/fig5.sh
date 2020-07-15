#!/usr/bin/env bash

cd ../code

python3 -m ESN_code.plotting.plot_performance_sweep_composite_R_t_local_het --hide_plot

cd ../plots

for file in hom_regulation_performance_sweep_composite_het.*
do
    cp $file ./hires/$file
    convert $file -resize 1000 $file
done
