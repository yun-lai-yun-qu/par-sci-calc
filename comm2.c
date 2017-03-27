#include "mpi.h"
#include <stdio.h>

int main (int argc, char **argv){

        MPI_Init(&argc,&argv);
        int size,rank, message,i,next,prev;
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        for (i=0;i<size;i++){
                message = rank*rank;
                next = rank + 1;
                prev = rank - 1;
                if (rank == size-1) next = 0;
                if (rank == 0) prev = size-1;
                MPI_Send(&message,1,MPI_INT,next,0,MPI_COMM_WORLD);
                MPI_Recv(&message,1,MPI_INT,prev,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        printf("%d di %d: il messaggio ricevuto e' %d\n",rank,size,message);
        MPI_Finalize();
        return 0;

}
