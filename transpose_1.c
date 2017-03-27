#include "mpi.h"
#include <stdio.h>

int main (int argc, char **argv){

        MPI_Init(&argc,&argv);
        int rank, size, row[4], col[4], i;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm_size(MPI_COMM_WORLD,&size);

        if (size != 4)
                return 0;

        for (i=0; i<4; i++)
                row[i]=4*rank + i;

        MPI_Alltoall(row,1,MPI_INT,col,1,MPI_INT,MPI_COMM_WORLD);

        printf("%d of %d: %d %d %d %d\n", rank, size, col[0], col[1], col[2], col[3]);

        MPI_Finalize();
        return 0;

}
