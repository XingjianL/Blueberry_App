#!/bin/bash python
echo "start script to process all images"
TOTAL_IMGS=$(ls drive-download-20220904T003514Z-001| wc -l)
#TOTAL_IMGS=$(ls Fairhope_052322| wc -l)
echo $TOTAL_IMGS
for i in $(seq 1 $TOTAL_IMGS)
do
    echo "current" $i
    ######python blueberry_all.py $i
    #python blueberry_final.py $i
    python ml.py $i
done