!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -pe orte 2
mpirun -np 8 ./pi_time 10000
