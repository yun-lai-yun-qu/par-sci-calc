#include "mpi.h"
#include <stdio.h>

int main (int argc, char **argv){

        MPI_Init(&argc,&argv);
        int rank, message;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        if (rank == 0){
                message = 1993;
                MPI_Send(&message,1,MPI_INT,1,3,MPI_COMM_WORLD);
        }
        if (rank == 1){
                MPI_Recv(&message,1,MPI_INT,0,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                printf("Il messaggio ricevuto e' %d\n",message);
        }
        MPI_Finalize();
        return 0;

}
