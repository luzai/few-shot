#!/bin/bash

#cd /mnt/SSD/luzai/imagenet22k-raw

find . -maxdepth 1 -size 10k -name '*.tar'  -exec  rm  {} \;
find . -maxdepth 1 -size 0  -name '*.tar'  -exec  rm  {} \;

for var  in $(ls /mnt/nfs1703/kchen/imagenet-raw/) 
do 
	echo $var
	var2=$(echo $var| sed -nE 's/(.*)\.tar/\1/p') 
	echo $var2
	if [ ! -d $var2 ]; then 
		mkdir $var2 
		tar xf $var2.tar -C $var2
		if [ ! $? -eq 0 ]; then 
			rm $var 
			rm $var2 -r
			echo "rm $var $var2 now"
        fi
    fi
done

#for var in $(find . -maxdepth 1 -type d)
#do 
#	ln -s /mnt/SSD/luzai/imagenet22k-raw/$var /mnt/nfs1703/kchen/imagenet-raw/ 
#done
