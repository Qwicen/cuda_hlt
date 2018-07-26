#!/bin/bash

cd build
rm test.txt

# measure performance for different chunks of 500 events
#for i in {0..9500..500}
#do
#    ./cu_hlt -f ../input/minbias/10kevents/velopix_raw/ -d ../input/minbias/10kevents/TrackerDumper/ -e ../input/minbias/10kevents/ut_hits/ -g ../input/geometry/ -n 500 -o $i -t 1 -r 100 -c 0 -v2 -m 300
#done
 
# measure statistical fluctuation of performance
for i in {0..9..1}                                                                                                                                                                                    
do                                                                                                                                                                                                         
    ./cu_hlt -f ../input/minbias/10kevents/velopix_raw/ -d ../input/minbias/10kevents/TrackerDumper/ -e ../input/minbias/10kevents/ut_hits/ -g ../input/geometry/ -n 10000 -t 4 -r 100 -c 0 -v2 -m 3300
done  
