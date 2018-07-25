#!/bin/bash

cd build
rm test.txt

for i in {0..9500..500}
do
    ./cu_hlt -f ../input/minbias/10kevents/velopix_raw/ -d ../input/minbias/10kevents/TrackerDumper/ -e ../input/minbias/10kevents/ut_hits/ -g ../input/geometry/ -n 500 -o $i -t 1 -r 100 -c 0 -v2 -m 300
done
