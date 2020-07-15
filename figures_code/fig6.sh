#!/usr/bin/env bash

cd ../code

python3 -m ESN_code.plotting.plot_rec_mem_pot_var_predict_R_a_fixed_composite --hide_plot

cd ../plots

for file in var_predict_composite_R_a_fixed.*
do
    cp $file ./hires/$file
    convert $file -resize 1000 $file
done
