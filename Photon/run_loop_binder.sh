#!/bin/sh

NAME=Core/Binderanalysis.py

NUM_THREADS=60
END_TASK_NO=1
ID=0

G_S=(0.1 0.2 0.3 0.5 0.6)
L_S=(8 16 32 48 64 96)
N=$1

if [ $(( $N )) == 2 ]
then
    B_S=(0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59)
fi

if [ $(( $N )) == 3 ]
then
    B_S=(0.40 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48)
fi

if [ $(( $N )) == 4 ]
then
    B_S=(0.42 0.43 0.44 0.45 0.46 0.47)
fi

LEN_G=${#G_S[@]}
LEN_L=${#L_S[@]}
LEN_B=${#B_S[@]}
END=$(( $LEN_G * $LEN_L * $LEN_B))

while [ $ID -le $END ]; do

  if [ $(( $RUNNING_PROCESSES )) -lt $(( $NUM_THREADS )) ]
  then
    G_INDEX=$(( ID/(LEN_B*LEN_L) ))
    L_INDEX=$(( (ID/LEN_B)%LEN_L ))
    B_INDEX=$(( ID%LEN_B ))
    G=${G_S[$G_INDEX]}
    L=${L_S[$L_INDEX]}
    B=${B_S[$B_INDEX]}

    echo about to run with N=$N ID=$ID G=$G L=$L B=$B

    python3 $NAME $N $G $B $L &

    ID=$(($ID + 1))
  fi

  RUNNING_PROCESSES=$(ps -ef | grep $NAME | wc -l)

  sleep 0.01
  
  echo RUNNING_PROCESSES : $RUNNING_PROCESSES
done


# Now we need to wait for all of the processes to finsish
while [ $RUNNING_PROCESSES -gt $END_TASK_NO ]; do
  
  sleep 0.5
  
  # RUNNING_PROCESSES=$(ps -ef | grep $NAME | egrep -v "grep|vi|more|pg" | wc -l)
  RUNNING_PROCESSES=$(ps -ef | grep $NAME | wc -l)

  echo RUNNING_PROCESSES : $RUNNING_PROCESSES
done
