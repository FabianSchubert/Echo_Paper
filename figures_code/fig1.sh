#!/usr/bin/env bash

cd ../code

python3 -m ESN_code.plotting.plot_alt_hom_regulation_composite --hide_plot

cd ../plots

for file in alt_hom_regulation_composite.*
do
    cp $file ./hires/$file
    convert $file -resize 1000 $file
done
