#!/usr/bin/env bash
files=$(find -maxdepth 1 -type d -name 'n*')
var=0
for file in $files; do 
    echo $file
    echo $var
    mv $file ../imagenet-raw-trans-to-redis/
    let var++
    if [ $var -ge 10 ]; then
        break 
    fi
done


