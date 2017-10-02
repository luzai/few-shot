rm -rf size.txt
#for i in $(find ./n10520286 -name '*.JPEG'); do                            
for i in $(find . -name '*.JPEG'); do                 
    var=$(identify $i | sed -nE 's/.*JPEG JPEG (.*)x([0-9]+) .*/\1 \2/p')
    echo $var
    echo $var >> size.txt

done

