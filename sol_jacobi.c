#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>

int main(argc,argv)
int argc;
char *argv[];
{
    int rank,mpi_size;
    int n_loc,n_loc_0,n;
    int it,nit_max;
    int i,ind;
    double *x_loc,*x_loc_1,*x;
    double *f_loc,*f;
    double a,b,h;
    float alpha,beta;
    double ys,yr;
    double err,*diff,max_diff,my_err;
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank==0){
            printf("Enter the number of local intervals: ");
            scanf("%d",&n_loc);
            printf("Enter u(0): ");
            scanf("%g",&alpha);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==mpi_size-1){
            printf("Enter u(1): ");
            scanf("%g",&beta);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&n_loc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n_loc_0=n_loc;

    n=n_loc*mpi_size;
 
    n_loc++;
    if(rank<mpi_size-1)
      n_loc++;
    n++;

    x_loc=(double *) malloc(n_loc*sizeof(double));
    x_loc_1=(double *) malloc(n_loc*sizeof(double));
    diff=(double *) malloc(n_loc*sizeof(double));
    f_loc=(double *) malloc(n_loc*sizeof(double));
    for(i=0;i<n_loc;i++){
      x_loc[i]=0.0; 
      x_loc_1[i]=0.0;  
      diff[i]=0.0;
      f_loc[i]=0.0;
    }

    if(rank==0)
      x_loc[0]=alpha;
    if(rank==mpi_size-1)
      x_loc[n_loc-1]=beta;

    h=1.0/(n-1.0);
    a=2.0/(h*h);
    b=-1.0/(h*h);

    x=(double *) malloc(n*sizeof(double));
    f=(double *) malloc(n*sizeof(double));
    for(i=0;i<n;i++){
      x[i]=0.0;
      f[i]=(i*h)*(i*h);
    }

//    for(i=0;i<n;i++)
//       printf("f(i) = %g\n",f[i]);

    if(rank==0){
      for(i=0;i<n_loc;i++)
        f_loc[i]=f[i];
      for(i=1;i<mpi_size;i++){
        ind=i*n_loc_0+1;
        MPI_Send(&f[ind],n_loc_0,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
      }
    }
    else
        MPI_Recv(&f_loc[1],n_loc_0,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&status);

//    for(i=0;i<n_loc;i++)
//       printf("f_loc(i) = %g\n",f_loc[i]);

    nit_max=1000000;
    it=0;
    err=1;

    // START JACOBI LOOP
    while(err>0.00000001&it<nit_max){
       if(rank<mpi_size-1){
          ys=x_loc[n_loc-2];
          MPI_Send(&ys,1,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);
       }
       if(rank>0){
          MPI_Recv(&yr,1,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,&status);          
          x_loc[0]=yr;
       }
       //MPI_Barrier(MPI_COMM_WORLD);

       if(rank>0){
          ys=x_loc[1];
          MPI_Send(&ys,1,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD);
       }
       if(rank<mpi_size-1){
          MPI_Recv(&yr,1,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,&status);
          x_loc[n_loc-1]=yr;
       }
       //MPI_Barrier(MPI_COMM_WORLD);

       x_loc_1[0]=x_loc[0];
       x_loc_1[n_loc-1]=x_loc[n_loc-1];

       for(i=1;i<n_loc-1;i++)
         x_loc_1[i]=1/a*(f_loc[i]-b*(x_loc[i-1]+x_loc[i+1]));

       my_err=0.0;

       for(i=0;i<n_loc;i++){
         diff[i]=fabs(x_loc_1[i]-x_loc[i]);
         if(diff[i]>my_err)
           my_err=diff[i];
         x_loc[i]=x_loc_1[i];
       }

       MPI_Allreduce(&my_err,&err,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

//       if(rank==0)
//         printf("err = %g\n",err);

       it++;

       //MPI_Barrier(MPI_COMM_WORLD);

    }
    // END JACOBI LOOP

//    if(rank>0)
//      MPI_Send(&x_loc[1],n_loc_0,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
//    else{
//      for(i=1;i<mpi_size;i++){
//         ind=i*n_loc_0+1;
//         MPI_Recv(&x[ind],n_loc_0,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&status); 
//      }
//    }
    
    ind=rank*n_loc_0+1;
    MPI_Gather(&x_loc[1],n_loc_0,MPI_DOUBLE,&x[ind],n_loc_0,MPI_DOUBLE,0,MPI_COMM_WORLD);

    // OUTPUT
    if(rank==0){
      x[0]=alpha;
      for(i=1;i<=n_loc_0;i++)
        x[i]=x_loc[i];
      for(i=0;i<n;i++)
        printf("x(%d) = %g\n",i,x[i]);
      printf("Jacobi converges in %d iterations\n",it);
    }

    MPI_Finalize();
    return 0;
}
