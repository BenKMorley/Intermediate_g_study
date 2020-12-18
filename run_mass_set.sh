#!/usr/bin/env bash

N=2
g=1
L=16
NOR=10
ALPHA=0

# Don't bother saving at the moment as the runs are so cheap
SAVE_FREQ=99999999

END=100000

NODES=1

# Use my virtual environment for this project
source /home/dc-kitc1/virtual_envs/Intermediate_g_env/bin/activate

python3 mass_array.py $N $g $L

# Leave this virtual environment
deactivate

X=`cat masses_to_run_temp.txt`

rm masses_to_run_temp.txt

cd /rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-hbor

for m in `echo $X`
do
    echo About to submit job with N=$N, g=$g, L=$L, m=$m

    ./start-run.sh $g $N $L $m
    sleep 1
done

