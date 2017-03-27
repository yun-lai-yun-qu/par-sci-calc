#!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -pe orte 2
mpirun -np 4 ./main
