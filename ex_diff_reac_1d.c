#include <stdio.h>
#include <math.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscpc.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt rank,size;
  PetscInt i,N,Nloc;
  PetscInt kpm1[2];
  PetscScalar h;
  PetscScalar one = 1.0;
  Vec uu,xx;
  Vec ff,bb;
  Vec uex;
  Vec err_u;
  DM da1d; 
  Mat AA,MM;
  Mat AA1;
  KSP ksp;
  PC pc;
  int loc_size;
  int its;
  double Aloc[2][2];
  double Mloc[2][2];
  double value;
  double *xl;
  double *fl;
  double *uel;
  double PI25DT=3.141592653589793238462643;
  double err_h1,err_l2;

  PetscInitialize(&argc,&argv,(char *)0,(char *)0);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  Nloc=16;
  N=size*Nloc;
  h=1.0/N;

  ierr=DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,N+1,1,1,NULL,&da1d);CHKERRQ(ierr);

  ierr=DMCreateGlobalVector(da1d,&uu);CHKERRQ(ierr);
  ierr=VecDuplicate(uu,&xx);CHKERRQ(ierr);
  ierr=DMCreateMatrix(da1d,MATMPIAIJ,&AA);CHKERRQ(ierr);
  ierr=DMCreateMatrix(da1d,MATMPIAIJ,&MM);CHKERRQ(ierr);

  Aloc[0][0]=1.0/h;
  Aloc[0][1]=-1.0/h;
  Aloc[1][0]=-1.0/h;
  Aloc[1][1]=1.0/h;

  Mloc[0][0]=h/3;
  Mloc[0][1]=h/6;
  Mloc[1][0]=h/6;
  Mloc[1][1]=h/3;

  for (i=0;i<Nloc;i++){
    kpm1[0]=i;
    kpm1[1]=i+1;

    ierr=MatSetValuesLocal(AA,2,kpm1,2,kpm1,Aloc[0],ADD_VALUES);CHKERRQ(ierr);
    ierr=MatSetValuesLocal(MM,2,kpm1,2,kpm1,Mloc[0],ADD_VALUES);CHKERRQ(ierr);    
  }
  ierr=MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr=MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr=MatAssemblyBegin(MM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr=MatAssemblyEnd(MM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr=VecSet(uu,1.0);CHKERRQ(ierr);
  ierr=MatMult(AA,uu,xx);CHKERRQ(ierr);
  ierr=VecDot(uu,xx,&value);
 
  if(rank==0){
    printf("check zeros AA %f\n",value);
  }

  ierr=DMCreateMatrix(da1d,MATMPIAIJ,&AA1);CHKERRQ(ierr);
  ierr=MatCopy(AA,AA1,SAME_NONZERO_PATTERN);

  ierr=VecSet(uu,1.0);CHKERRQ(ierr);
  ierr=MatMult(MM,uu,xx);CHKERRQ(ierr);
  ierr=VecDot(uu,xx,&value);

  if(rank==0){
    printf("volume = %f\n",value);
  }

  ierr=VecDuplicate(uu,&ff);CHKERRQ(ierr);
  ierr=VecDuplicate(uu,&bb);CHKERRQ(ierr);
  ierr=VecDuplicate(uu,&uex);CHKERRQ(ierr);

  if(rank==0){
    for(i=0;i<N+1;i++){
      value=h*i;
      ierr=VecSetValues(xx,1,&i,&value,INSERT_VALUES);
    }
  }
  ierr=VecAssemblyBegin(xx);CHKERRQ(ierr);
  ierr=VecAssemblyEnd(xx);CHKERRQ(ierr);

//  ierr=VecView(xx,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr=VecGetLocalSize(xx,&loc_size);CHKERRQ(ierr);

  ierr=VecGetArray(xx,&xl);CHKERRQ(ierr);
  ierr=VecGetArray(ff,&fl);CHKERRQ(ierr);
  ierr=VecGetArray(uex,&uel);CHKERRQ(ierr); 
  for(i=0;i<loc_size;i++){
//    fl[i]=1.0;
      fl[i]=(pow(PI25DT,2)+1.0)*cos(PI25DT*xl[i]);
      uel[i]=cos(PI25DT*xl[i]);
  }
  ierr=VecRestoreArray(xx,&xl);CHKERRQ(ierr);
  ierr=VecRestoreArray(ff,&fl);CHKERRQ(ierr);
  ierr=VecRestoreArray(uex,&uel);CHKERRQ(ierr);
 
//  ierr=VecView(ff,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr=MatMult(MM,ff,bb);CHKERRQ(ierr);

//  ierr=VecView(bb,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr=MatAXPY(AA,1.0,MM,SAME_NONZERO_PATTERN);

// Create linear solver KSP
  ierr=KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr=KSPSetOperators(ksp,AA,AA,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr=KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr=KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr=KSPSetTolerances(ksp,1.e-6,1.e-40,1.e5,100000);CHKERRQ(ierr);
  ierr=KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr=PCSetType(pc,PCBJACOBI);CHKERRQ(ierr);
  ierr=KSPSetFromOptions(ksp);CHKERRQ(ierr);

// Solve the linear system
  ierr=VecSet(uu,0.0);CHKERRQ(ierr);
  ierr=KSPSolve(ksp,bb,uu);CHKERRQ(ierr);
  ierr=KSPGetIterationNumber(ksp,&its);

// Compute error
  ierr=VecDuplicate(uu,&err_u);CHKERRQ(ierr);
  ierr=VecCopy(uu,err_u);CHKERRQ(ierr);
  ierr=VecAXPY(err_u,-1.0,uex);CHKERRQ(ierr);
  
  ierr=MatMult(AA1,err_u,xx);CHKERRQ(ierr);
  ierr=VecDot(err_u,xx,&err_h1);
  err_h1=sqrt(err_h1);

  ierr=MatMult(MM,err_u,xx);CHKERRQ(ierr);
  ierr=VecDot(err_u,xx,&err_l2);
  err_l2=sqrt(err_l2);

  if(rank==0){
    printf("CG iterations = %d\n",its);
    printf("err H1 = %f\n",err_h1);
    printf("err L2 = %f\n",err_l2);
  }

  ierr=VecView(uu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
 
