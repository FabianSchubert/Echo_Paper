#!/usr/bin/env bash

cd ../code

python3 -m ESN_code.plotting.plot_performance_sweep_composite_R_t_local_hom --hide_plot

cd ../plots

for file in performance_sweep_composite_R_t_local_hom.*
do
    cp $file ./hires/$file
    convert $file -resize 1000 $file
done
