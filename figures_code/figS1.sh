#!/usr/bin/env bash

cd ../code

python3 -m ESN_code.plotting.plot_lyap_exp_conv --hide_plot

cd ../plots

for file in lyap_exp_conv.*
do
    cp $file ./hires/$file
    convert $file -resize 1000 $file
done
