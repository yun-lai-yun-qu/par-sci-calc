#include <stdio.h>
#include <math.h>
#include "mpi.h"
int main (int argc,char **argv){
        MPI_Init(&argc,&argv);

        double starttime, endtime, mytime, maxtime, mintime, avgtime;
        starttime = MPI_Wtime();
        int size,rank;
        int n = 4;
        if (argc>0)
                n=atoi(argv[1]);
        else
                n=0;
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        double h=1.0/(size*n);
        double c=0.0,x,tot_sum,PI=3.141592653;
        int i;
        for(i=rank*n;i<(rank+1)*n;i++){
                x=h*i+h/2;
                c=c+h*4.0/(1+(x*x));
        }
        printf("PI, partial result - %d of %d: %f\n",rank,size,c);

        // invio di tutti i dati ad un solo processore
        MPI_Reduce(&c,&tot_sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

        endtime = MPI_Wtime();
        mytime = endtime - starttime;
        printf("Time - %d of % d, my time is %g\n",rank,size,mytime);
        MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mytime, &mintime, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
        MPI_Reduce(&mytime, &avgtime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
if (rank==0) {
                printf("PI, final result: %d of %d: %f, err:%lf\n",rank,size,tot_sum,fabs(tot_sum-PI));
                avgtime /= size;
                printf("Time - Min: %lf Max: %lf Avg: %lf\n", mintime, maxtime,avgtime);
        }
        MPI_Finalize();
        return 0;
}
