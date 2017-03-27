#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>

int main(argc,argv)
int argc;
char *argv[];
{
    int rank,numprocs;
    int n,nb;
    int i,j,k;

    double *sb,*rb;
    double **A,**At;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    n=10;
    nb=n*numprocs;
 
    A=malloc(n*sizeof(double *));
    At=malloc(n*sizeof(double *));
    for(i=0;i<n;i++){
       A[i]=malloc(nb*sizeof(double));
       At[i]=malloc(nb*sizeof(double)); 
    } 

    for(i=0;i<n;i++){
      for(j=0;j<nb;j++) A[i][j]=i*nb+j+rank*n*nb;
    }

//    if(rank==3){
//      for(i=0;i<n;i++){
//         printf("row = %d\n",i+1);
//         for(j=0;j<nb;j++) printf("%lf\n",A[i][j]);
//      }
//    }

    sb=(double *) malloc(nb*sizeof(double));
    rb=(double *) malloc(nb*sizeof(double));

    for(k=0;k<n;k++){
 
    for(j=0;j<numprocs;j++){
       for(i=0;i<n;++i){
          sb[j*n+i]=A[i][j*n+k];
       }
    }

    MPI_Alltoall(sb,n,MPI_DOUBLE,rb,n,MPI_DOUBLE,MPI_COMM_WORLD);

//    if(rank==2){
//       for(i=0;i<nb;i++){
//          printf("%lf\t", rb[i]);
//       }
//    }

    for(i=0;i<nb;++i){
       At[k][i]=rb[i];
    }

    }

    if(rank==3){
      for(i=0;i<n;i++){
         printf("row = %d\n",i+1);
         for(j=0;j<nb;j++) printf("%lf\n",At[i][j]);
      }
    }

    MPI_Finalize();
    return 0;
}
