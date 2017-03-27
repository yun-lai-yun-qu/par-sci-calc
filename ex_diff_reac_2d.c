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
  PetscInt i,j;
  PetscInt iindex;
  PetscInt ni_l,nj_l;
  PetscInt ni,nj;
  PetscInt ni1,nj1;
  PetscInt procx,procy;
  PetscInt kpm1[4];
  PetscInt gii,gjj;
  PetscInt gim,gjm;
  PetscScalar h;
  PetscScalar one=1.0;
  PetscScalar alpha,beta;
  Vec uu;
  Vec xx,yy;
  Vec xx_n;
  Vec ff,bb;
  Vec cc,cc1;
  Vec uex;
  Vec err_u;
  DM da2d; 
  Mat AA,MM;
  Mat AA1;
  KSP ksp;
  PC pc;
  int loc_size;
  int its;
  double Aloc[4][4];
  double Mloc[4];
  double value;
  double *xl,*yl;
  double *fl;
  double *uel;
  double PI25DT=3.141592653589793238462643;
  double err_h1,err_l2;

  PetscInitialize(&argc,&argv,(char *)0,(char *)0);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ni_l=32;
  nj_l=32;

  ierr=PetscOptionsGetInt(PETSC_NULL,"-nxl",&ni_l,PETSC_NULL);CHKERRQ(ierr);
  ierr=PetscOptionsGetInt(PETSC_NULL,"-nyl",&nj_l,PETSC_NULL);CHKERRQ(ierr);

  procx=2;
  procy=2;

  ierr=PetscOptionsGetInt(PETSC_NULL,"-px",&procx,PETSC_NULL);CHKERRQ(ierr);
  ierr=PetscOptionsGetInt(PETSC_NULL,"-py",&procy,PETSC_NULL);CHKERRQ(ierr);

  ni=procx*ni_l;
  nj=procy*nj_l;
  ni1=ni+1;
  nj1=nj+1;

  h=1.0/ni;

  ierr=DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,ni1,nj1,procx,procy,1,1,NULL,NULL,&da2d);CHKERRQ(ierr);

  ierr=DMCreateGlobalVector(da2d,&uu);CHKERRQ(ierr);
  ierr=VecDuplicate(uu,&xx);CHKERRQ(ierr);
  ierr=DMCreateMatrix(da2d,MATMPIAIJ,&AA);CHKERRQ(ierr);
  ierr=VecDuplicate(uu,&cc);CHKERRQ(ierr);

  ierr=DMDAGetGhostCorners(da2d,&gii,&gjj,NULL,&gim,&gjm,NULL);CHKERRQ(ierr);
  printf("rank = %d\t gii = %d\t gjj = %d\t gim = %d\t gjm = %d\n",rank,gii,gjj,gim,gjm);

  Aloc[0][0]=4.0/6.0;
  Aloc[0][1]=-1.0/6.0;
  Aloc[0][2]=-1.0/6.0;
  Aloc[0][3]=-2.0/6.0;

  Aloc[1][0]=-1.0/6.0;
  Aloc[1][1]=4.0/6.0;
  Aloc[1][2]=-2.0/6.0;
  Aloc[1][3]=-1.0/6.0;

  Aloc[2][0]=-1.0/6.0;
  Aloc[2][1]=-2.0/6.0;
  Aloc[2][2]=4.0/6.0;
  Aloc[2][3]=-1.0/6.0;

  Aloc[3][0]=-2.0/6.0;
  Aloc[3][1]=-1.0/6.0;
  Aloc[3][2]=-1.0/6.0;
  Aloc[3][3]=4.0/6.0;

  Mloc[0]=h*h/4;
  Mloc[1]=h*h/4;
  Mloc[2]=h*h/4;
  Mloc[3]=h*h/4;

  for (i=0;i<ni_l;i++){
   for (j=0;j<nj_l;j++){
    kpm1[0]=j*gim+i;
    kpm1[1]=j*gim+i+1;
    kpm1[2]=(j+1)*gim+i;
    kpm1[3]=(j+1)*gim+i+1;

    ierr=MatSetValuesLocal(AA,4,kpm1,4,kpm1,Aloc[0],ADD_VALUES);CHKERRQ(ierr);
    ierr=VecSetValuesLocal(cc,4,kpm1,Mloc,ADD_VALUES);CHKERRQ(ierr);    
   }
  }
  ierr=MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr=MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr=VecAssemblyBegin(cc);CHKERRQ(ierr);
  ierr=VecAssemblyEnd(cc);CHKERRQ(ierr);

  ierr=VecSet(uu,1.0);CHKERRQ(ierr);
  ierr=MatMult(AA,uu,xx);CHKERRQ(ierr);
  ierr=VecDot(uu,xx,&value);
 
  if(rank==0){
    printf("check zeros AA %f\n",value);
  }

  ierr=DMCreateMatrix(da2d,MATMPIAIJ,&AA1);CHKERRQ(ierr);
  ierr=MatCopy(AA,AA1,SAME_NONZERO_PATTERN);
  alpha=1.0;
  ierr=MatScale(AA,alpha);CHKERRQ(ierr);
  beta=1.0;
  ierr=VecDuplicate(cc,&cc1);CHKERRQ(ierr);
  ierr=VecCopy(cc,cc1);CHKERRQ(ierr);
  ierr=VecScale(cc1,beta);CHKERRQ(ierr);
  ierr=MatDiagonalSet(AA,cc1,ADD_VALUES);CHKERRQ(ierr);
  ierr=VecDestroy(&cc1);

  ierr=VecSet(uu,1.0);CHKERRQ(ierr);
  ierr=VecDot(uu,cc,&value);

  if(rank==0){
    printf("volume = %f\n",value);
  }

  ierr=VecDuplicate(uu,&ff);CHKERRQ(ierr);
  ierr=VecDuplicate(uu,&bb);CHKERRQ(ierr);
  ierr=VecDuplicate(uu,&uex);CHKERRQ(ierr);

  ierr=DMDACreateNaturalVector(da2d,&xx_n);CHKERRQ(ierr);
  ierr=VecSet(xx_n,0.0);CHKERRQ(ierr);
  ierr=VecSet(xx,0.0);CHKERRQ(ierr);
  if(rank==0){
    for(i=0;i<ni1;i++){
     for(j=0;j<nj1;j++){
      iindex=j*ni1+i;
      value=h*i;
      ierr=VecSetValues(xx_n,1,&iindex,&value,INSERT_VALUES);
     }
    }
  }
  ierr=VecAssemblyBegin(xx_n);CHKERRQ(ierr);
  ierr=VecAssemblyEnd(xx_n);CHKERRQ(ierr);
  ierr=DMDANaturalToGlobalBegin(da2d,xx_n,INSERT_VALUES,xx);CHKERRQ(ierr);
  ierr=DMDANaturalToGlobalEnd(da2d,xx_n,INSERT_VALUES,xx);CHKERRQ(ierr);

  ierr=VecSet(xx_n,0.0);CHKERRQ(ierr);
  ierr=VecDuplicate(xx,&yy);CHKERRQ(ierr);
  ierr=VecSet(yy,0.0);CHKERRQ(ierr);
  if(rank==0){
    for(i=0;i<ni1;i++){
     for(j=0;j<nj1;j++){
      iindex=j*ni1+i;
      value=h*j;
      ierr=VecSetValues(xx_n,1,&iindex,&value,INSERT_VALUES);
     }
    }
  }
  ierr=VecAssemblyBegin(xx_n);CHKERRQ(ierr);
  ierr=VecAssemblyEnd(xx_n);CHKERRQ(ierr);
  ierr=DMDANaturalToGlobalBegin(da2d,xx_n,INSERT_VALUES,yy);CHKERRQ(ierr);
  ierr=DMDANaturalToGlobalEnd(da2d,xx_n,INSERT_VALUES,yy);CHKERRQ(ierr);

//  ierr=VecView(xx,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
//  ierr=VecView(yy,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr=VecGetLocalSize(xx,&loc_size);CHKERRQ(ierr);

  ierr=VecGetArray(xx,&xl);CHKERRQ(ierr);
  ierr=VecGetArray(yy,&yl);CHKERRQ(ierr);
  ierr=VecGetArray(ff,&fl);CHKERRQ(ierr);
  ierr=VecGetArray(uex,&uel);CHKERRQ(ierr); 
  for(i=0;i<loc_size;i++){
//    fl[i]=1.0;
      fl[i]=(2.0*pow(PI25DT,2)+1.0)*cos(PI25DT*xl[i])*cos(PI25DT*yl[i]);
      uel[i]=cos(PI25DT*xl[i])*cos(PI25DT*yl[i]);
  }
  ierr=VecRestoreArray(xx,&xl);CHKERRQ(ierr);
  ierr=VecRestoreArray(yy,&yl);CHKERRQ(ierr);
  ierr=VecRestoreArray(ff,&fl);CHKERRQ(ierr);
  ierr=VecRestoreArray(uex,&uel);CHKERRQ(ierr);
 
//  ierr=VecView(ff,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr=VecPointwiseMult(bb,cc,ff);CHKERRQ(ierr);

//  ierr=VecView(bb,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

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

  ierr=VecPointwiseMult(xx,cc,err_u);CHKERRQ(ierr);
  ierr=VecDot(err_u,xx,&err_l2);
  err_l2=sqrt(err_l2);

  if(rank==0){
    printf("CG iterations = %d\n",its);
    printf("err H1 = %f\n",err_h1);
    printf("err L2 = %f\n",err_l2);
  }

//  ierr=VecView(uu,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
 
