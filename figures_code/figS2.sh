#!/usr/bin/env bash

cd ../code

python3 -m ESN_code.plotting.plot_alt_hom_regulation_performance_sweep_composite_hom --hide_plot

cd ../plots

for file in alt_hom_regulation_performance_sweep_composite_hom.*
do
    cp $file ./hires/$file
    convert $file -resize 1000 $file
done
