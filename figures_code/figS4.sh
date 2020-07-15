#!/usr/bin/env bash

cd ../code

python3 -m ESN_code.plotting.plot_corr_R_a_hom --hide_plot

cd ../plots

for file in corr_R_hom.*
do
    cp $file ./hires/$file
    convert $file -resize 1000 $file
done