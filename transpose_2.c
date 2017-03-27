#include "mpi.h"
#include <stdio.h>

int main (int argc, char **argv){

        MPI_Init(&argc,&argv);
        int N, k , i, j, rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm_size(MPI_COMM_WORLD,&size);

        if (argc>0)
                N=atoi(argv[1]);
        else
                N=0;

        if ((N % size) != 0)
                return 0;

        int Nloc = N / size;
        int mat[Nloc][N];
        int col[N], row[N];

        for (i=0; i < Nloc; i++)
                for (j=0; j < N; j++)
                        mat[i][j] = N*i + j;

/*      if (rank == 0) {
                for (i=0; i<N; i++)
                        printf("%d ",mat[0][i]);
                printf("\n");
        }
*/

        for (k=0; k < Nloc; k++) {
                for (j=0; j < size; j++)
                        for (i=0; i < Nloc; i++)
                                col[i+Nloc*j] = mat[i][j*Nloc+k];
                MPI_Alltoall(col,Nloc,MPI_INT,row,Nloc,MPI_INT,MPI_COMM_WORLD);

                if (rank == 0) {
                        for (i=0; i<N; i++)
                                printf("%d ",row[i]);
                        printf("\n");
                }
        }

        MPI_Finalize();
        return 0;

}
