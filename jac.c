#include "mpi.h"
#include <stdio.h>

int main (int argc, char **argv){

        // risoluzione approssimata di -delta(u) = f con metodo di Jacobi

        MPI_Init(&argc,&argv);
        int rank, size, message, n, N, i, j, iter = 0, maxiter = 10000;
        double a = 0, b = 1, ua = 0, ub = 1, h, maxerr = 1e-8, err_pmax;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm_size(MPI_COMM_WORLD,&size);

        // n numero di nodi interni per processore(cioe' xa e xb non considerati)

        if (argc > 1) {
                n = atoi(argv[1]);
        }
        else return 0;

        if (a > b) return 0;
        N = size*n + 2;
        h = ((double)(b - a))/(size*n + 1);

        double x_p[n+2];
        double f_p[n+2];
        double u_pk[n+2];
        double u_pnew[n+2];
        double err_p[n];
        double err;

        for (i=1; i<=n; i++)
                x_p[i] = a + (rank*n + i)*h;
        if (rank == 0)
                x_p[0] = a;
        if (rank == size-1)
                x_p[n+1] = b;

        for (i=1; i<=n; i++)
                f_p[i] = 12*x_p[i]*x_p[i];
        if (rank == 0)
                f_p[0] = 12*x_p[0]*x_p[0];
        if (rank == size-1)
                f_p[n+1] = 12*x_p[n+1]*x_p[n+1];

        for (i=0; i<n+2; i++)
                u_pk[i] = 0;
        if (rank == 0) {
                u_pk[0] = ua;
                u_pnew[0] = ua;
        }
        if (rank == size-1) {
                u_pk[n+1] = ub;
                u_pnew[n+1] = ub;
        }

        while ((iter < maxiter)) {
                iter++;
                for (i=1; i<=n; i++)
                        u_pnew[i] = ((h*h)/2)*(u_pk[i-1]/(h*h) + u_pk[i+1]/(h*h) - f_p[i]);
                if (rank < size-1) {
                        MPI_Recv(&u_pnew[n+1],1,MPI_DOUBLE,rank+1,iter,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
                if (rank > 0) {
                        MPI_Send(&u_pnew[1],1,MPI_DOUBLE,rank-1,iter,MPI_COMM_WORLD);
                }
                if (rank > 0) {
                        MPI_Recv(&u_pnew[0],1,MPI_DOUBLE,rank-1,iter+1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
                if (rank < size-1) {
                        MPI_Send(&u_pnew[n],1,MPI_DOUBLE,rank+1,iter+1,MPI_COMM_WORLD);
                }

                err_pmax = 0;
                for (j=0; j<n; j++) {
                        err_p[j] = u_pnew[j]-u_pk[j];
                        if (err_p[j] < 0) err_p[j] = -err_p[j];
                        if (err_p[j] > err_pmax)
                                err_pmax = err_p[j];
                }
                MPI_Reduce(&err_pmax,err,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
                // comunicare err_max a P0
                // se rank == 0 calcolare il max degli err_pmax
                // fare un broadcast del massimo errore globale err
                // (aggiungera la condizione su err al ciclo while)

                for (i=0; i<=n+1; i++)
                        u_pk[i] = u_pnew[i];

        }

        if (rank == 0) {
                for (j=0; j<=n; j++)
                        printf("%f ",u_pk[j]);
                printf("\n");
                for (j=0; j<=n; j++)
                        printf("%f ",f_p[j]);
                printf("\n");
        }

        MPI_Finalize();
        return 0;

}
