
#include <stdio.h>
#include <math.h>
#include "mpi.h"
int main (int *argc,char **argv){
        MPI_Init(&argc,&argv);
        int size,rank;
        int n;
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
	printf("%d of %d: %f\n",rank,size,c);
        MPI_Reduce(&c,&tot_sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if (rank==0) printf("%d of %d: %f %lf\n",rank,size,tot_sum,fabs(tot_sum-PI));
        MPI_Finalize();
        return;
}

