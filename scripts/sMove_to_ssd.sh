#!/usr/bin/env bash
files=$(find -maxdepth 1 -type d -name 'n*')
var=0
for file in $files; do 
    echo $file
    echo $var
    cp $file /mnt/SSD/luzai/imagenet22k-raw/$file -r
    let var++
    if [ $var -ge 2000 ]; then
        break 
    fi
done


