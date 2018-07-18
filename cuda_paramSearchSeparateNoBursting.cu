#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda.h"
#include "curand.h"
#include "curand_kernel.h"
#include <time.h>

const int num_params = 1000000;
const int MAX_RNA = 50;
const int num_cells = 10000;
const int gene_length = 2100;//GAL1 1587, GAL10 2100, HMS2 1077, CMK2 1344, GFP 708bp

double MAX(double a,double b){
  if (a > b){return a;}
  else{return b;}
}

#include "file_funcs.h"//read_data
#include "sample_params.h"//choose_params,create_paramsVecs
#include "calculate_metrics.h"//calculate_CDF,calculate_ks
#include "cuda_randomFuncs.h"//setup_kernel,ran_gamma
#include "cuda_gillespieNoBursting.h"//sim_gillespie

int main(int argc,char* argv[]){
  double increment = 1.0/(double)num_cells;

  double** raw_data;
  raw_data = (double**)malloc(MAX_RNA * sizeof(double*));
  for (int i=0;i<MAX_RNA;i++){
    raw_data[i] = (double*)malloc(MAX_RNA * sizeof(double));
  }
  read_data(argv[1],raw_data);
  double tot_data = 0.0;
  for (int i=0;i<MAX_RNA;i++){
    for (int j=0;j<MAX_RNA;j++){
      tot_data += raw_data[i][j];
    }
  }
  for (int i=0;i<MAX_RNA;i++){
    for (int j=0;j<MAX_RNA;j++){
      raw_data[i][j] /= tot_data;
    }
  }
  double* raw_nuc;
  double* raw_cyt;
  double* raw_nucCDF;
  double* raw_cytCDF;
  raw_nuc = (double*)malloc(MAX_RNA * sizeof(double));
  raw_cyt = (double*)malloc(MAX_RNA * sizeof(double));
  raw_nucCDF = (double*)malloc(MAX_RNA * sizeof(double));
  raw_cytCDF = (double*)malloc(MAX_RNA * sizeof(double));
  for (int nuc=0;nuc<MAX_RNA;nuc++){
    raw_nuc[nuc]=0.0;
    for (int cyt=0;cyt<MAX_RNA;cyt++){
      raw_nuc[nuc]+=raw_data[nuc][cyt];
    }
    for (int i=nuc;i<MAX_RNA;i++){
      raw_nucCDF[i]+=raw_nuc[nuc];
    }
  }
  for (int cyt=0;cyt<MAX_RNA;cyt++){
    raw_cyt[cyt]=0.0;
    for (int nuc=0;nuc<MAX_RNA;nuc++){
      raw_cyt[cyt]+=raw_data[nuc][cyt];
    }
    for (int i=cyt;i<MAX_RNA;i++){
      raw_cytCDF[i]+=raw_cyt[cyt];
    }
  }
  free(raw_nuc);
  free(raw_cyt);
  free(raw_data);
  //Create vectors with parameters chosen from a latin-hypercube
  double* max_param;
  max_param = (double*)malloc(3 * sizeof(double));//Set maximum values of each parameter
  max_param[0] = 4.0;
  max_param[1] = 5.0;
  max_param[2] = 0.5;
  double** chosen_params;
  chosen_params = (double**)malloc(3 * sizeof(double*));
  for (int i=0;i<3;i++){
    chosen_params[i] = (double*)malloc(num_params * sizeof(double));
  }
  create_paramsVecsNoBurst(max_param,chosen_params);
  free(max_param);

  //Simulate distributions, calculate CDFs and KS statistics
  double* nuc_dist;
  nuc_dist = (double*)malloc(MAX_RNA * sizeof(double));
  double* cyt_dist;
  cyt_dist = (double*)malloc(MAX_RNA * sizeof(double));

  //Determine number of blocks and threads to use on GPU
  int num_blocks,min_GridSize,num_threads;
  cudaOccupancyMaxPotentialBlockSize(&min_GridSize,&num_threads,sim_gillespie,0,num_cells);
  num_blocks = (num_cells + num_threads - 1)/num_threads;

  //Setup random environment
  curandState *devStates;
  cudaMalloc((void**)&devStates,num_cells * sizeof(curandState));
  setup_kernel<<<num_blocks,num_threads>>>(devStates);

  FILE *fp;
  fp = fopen(argv[2],"w");
  fprintf(fp,"INIT\tELONG\tDEG\tKS_NUC\tKS_CYT\n");

  int* recv_nuc;
  recv_nuc = (int*)malloc(num_cells * sizeof(int));
  int* recv_cyt;
  recv_cyt = (int*)malloc(num_cells * sizeof(int));
  int* device_nuc;
  cudaMalloc((void**)&device_nuc,num_cells * sizeof(int));
  int* device_cyt;
  cudaMalloc((void**)&device_cyt,num_cells * sizeof(int));

  double* params;
  cudaMalloc((void**)&params,3 * sizeof(double));
  double* it_params;
  it_params = (double*)malloc(3 * sizeof(double));

  for (int iteration = 0;iteration<num_params;iteration++){
    for (int i=0;i<MAX_RNA;i++){
      nuc_dist[i] = 0.0;
      cyt_dist[i] = 0.0;
    }

    for (int i=0;i<3;i++){it_params[i] = chosen_params[i][iteration];}
    cudaMemcpy(params,it_params,3 * sizeof(double),cudaMemcpyHostToDevice);

    sim_gillespie<<<num_blocks,num_threads>>>(params,device_nuc,device_cyt,devStates);

    cudaMemcpy(recv_nuc,device_nuc,num_cells * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(recv_cyt,device_cyt,num_cells * sizeof(int),cudaMemcpyDeviceToHost);

    for (int cell=0;cell<num_cells;cell++){
      if ((recv_nuc[cell]<MAX_RNA)&&(recv_cyt[cell]<MAX_RNA)){
        nuc_dist[recv_nuc[cell]]+=increment;
        cyt_dist[recv_cyt[cell]]+=increment;
      }
    }
    double* sim_nucCDF;
    double* sim_cytCDF;
    sim_nucCDF=(double*)malloc(MAX_RNA * sizeof(double));
    sim_cytCDF=(double*)malloc(MAX_RNA * sizeof(double));
    double ks_nuc,ks_cyt;
    ks_nuc = calculate_1DKS(raw_nucCDF,nuc_dist,sim_nucCDF);
    ks_cyt = calculate_1DKS(raw_cytCDF,cyt_dist,sim_cytCDF);
    for (int i=0;i<3;i++){
      fprintf(fp,"%f\t",it_params[i]);
    }
    fprintf(fp,"%f\t%f\n",ks_nuc,ks_cyt);
  }
  fclose(fp);

  cudaFree(device_nuc);
  cudaFree(device_cyt);
  cudaFree(params);
  cudaFree(devStates);

  free(recv_nuc);
  free(recv_cyt);
  free(it_params);

  free(chosen_params);

  return 0;
}
