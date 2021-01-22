#!/usr/bin/env bash

N=$1
g=$2
L=$3
SPAN=$4
nOR=10
ALPHA=0.0001

# Don't bother saving at the moment as the runs are so cheap
FREQ=99999999

# One larger than end because of fencepost
TPJOB=100001

END=100000

NODES=1

# Use my virtual environment for this project
source /home/dc-kitc1/virtual_envs/Intermediate_g_env/bin/activate

python3 mass_array.py $N $g $L $SPAN

# Leave this virtual environment
deactivate

X=`cat masses_to_run_temp.txt`

rm masses_to_run_temp.txt

cd /rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-hbor-dbtest

for m in `echo $X`
do
    echo About to submit job with N=$N, g=$g, L=$L, m=$m

    echo ./start-run.sh $g $N $L $m $nOR $ALPHA $FREQ $TPJOB $END $NODES

    ./start-run.sh $g $N $L $m $nOR $ALPHA $FREQ $TPJOB $END $NODES
    sleep 1
done

