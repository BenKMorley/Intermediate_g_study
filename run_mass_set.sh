#!/usr/bin/env bash

N=2
g=1
L=16
NOR = 10
ALPHA = 0
SAVE_FREQ = 


END = 
NODES = 1

python3 mass_array.py $N $g $L

X=`cat masses_to_run_temp.txt`

rm masses_to_run_temp.txt

cd /rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-hbor

for m in `echo $X`
do
    echo About to submit job with N=$N, g=$g, L=$16, m=$m

    # ./do_start_run.sh $g $N $L $m 
    sleep 1
done

