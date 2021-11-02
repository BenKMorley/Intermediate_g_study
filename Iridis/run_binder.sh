#!/bin/sh
#SBATCH --job-name=serial_array   # Job name
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bkm1n18@soton.ac.uk # Where to send mail  
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1
#SBATCH --time=10:00:00             # Time limit hrs:min:sec
#SBATCH --output=log/double_evidence-%A-%a.out    # Standard output and error log
#SBATCH --array=0-279              # Array range

gs=(0.1 0.2 0.3 0.5 0.6)
Ls=(8 16 32 48 64 96 128)
Bbars=(0.40 0.41 0.42 0.43 0.44 0.45 0.46 0.47)
len_Bbars=8
len_Ls=7

SLURM_ARRAY_TASK_ID=279

div=$(( $len_Bbars * $len_Ls ))

g_index=$(( $SLURM_ARRAY_TASK_ID / $div ))
Rem=$(( $SLURM_ARRAY_TASK_ID % $div ))
L_index=$(( $Rem / $len_Bbars ))
Bbar_index=$(( $Rem % $len_Bbars ))

g=${gs[ $g_index ]}
L=${Ls[ $L_index ]}
Bbar=${Bbars[ $Bbar_index ]}

python3 Local/Binderanalysis.py $N $g $Bbar $L > log/Binderanalysis_log_N${N}_g${g}_Bbar${Bbar}_L${L}.out
