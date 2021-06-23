#!/bin/sh

NAME=Core/Binderanalysis.py
DIR=h5data/
FILENAME=MCMCdata.h5
THERM=0
# NBOOT=500


NUM_THREADS=30
END_TASK_NO=1
ID=0
N=4

G_S=(0.1 0.2 0.3 0.5 0.6)
L_S=(8 16 32 48 64 96)
B_S=(0.40 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48)

LEN_G=${#G_S[@]}
LEN_L=${#L_S[@]}
LEN_B=${#B_S[@]}
END=$(( $LEN_G * $LEN_L * $LEN_B ))
G_SIZE=$(( $END / $LEN_G ))
M=$(( $LEN_G * $LEN_L ))

# Run from inside photon directory
cd ..


while [ $ID -le $END ]; do

  if [ $(( $RUNNING_PROCESSES )) -lt $(( $NUM_THREADS )) ]
  then
    G_INDEX=$(( ID/(LEN_B*LEN_L) ))
    L_INDEX=$(( (ID/LEN_B)%LEN_L ))
    B_INDEX=$(( ID%LEN_B ))
    G=${G_S[$G_INDEX]}
    L=${L_S[$L_INDEX]}
    B=${B_S[$B_INDEX]}

    echo about to run with ID=$ID G=$G L=$L B=$B

    python3 $NAME $N $G $B $L -therm=$THERM -filename=$FILENAME &

    ID=$(($ID + 1))
  fi

  RUNNING_PROCESSES=$(ps -ef | grep $NAME | wc -l)

  sleep 0.5
  
  echo RUNNING_PROCESSES : $RUNNING_PROCESSES
done


# Now we need to wait for all of the processes to finsish
while [ $RUNNING_PROCESSES -gt $END_TASK_NO ]; do
  
  sleep 0.5
  
  # RUNNING_PROCESSES=$(ps -ef | grep $NAME | egrep -v "grep|vi|more|pg" | wc -l)
  RUNNING_PROCESSES=$(ps -ef | grep $NAME | wc -l)

  echo RUNNING_PROCESSES : $RUNNING_PROCESSES
done
