#!/usr/bin/env bash

cd ../code

python3 -m ESN_code.plotting.plot_alt_hom_regulation_flow heterogeneous_identical_binary --hide_plot

cd ../plots

for file in heterogeneous_identical_binary_input_alt_hom_regulation_flow.*
do
    cp $file ./hires/$file
    convert $file -resize 1000 $file
done
