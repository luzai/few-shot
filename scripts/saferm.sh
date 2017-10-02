for file in $(cat test); do 
	echo $file
	rm -rf $file 
done
