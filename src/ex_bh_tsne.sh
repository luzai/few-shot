#!/usr/bin/env bash
declare -a ps=("5" '35' 50 65 100)
#declare -a ps=("element1" "element2" "element3")

for p in "${ps[@]}"
do
        cd p.$p.dim.2
        ln -s ../bh_tsne .
        ./bh_tsne
        cd ..
        cd p.$p.dim.3
        ln -s ../bh_tsne .
        ./bh_tsne
        cd ..
done
