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
const int gene_length = 2100;//GAL1 1587, GAL10 2100

double MAX(double a,double b){
  if (a > b){return a;}
  else{return b;}
}

#include "file_funcs.h"//read_data
#include "sample_params.h"//choose_params,create_paramsVecs
#include "calculate_metrics.h"//calculate_CDF,calculate_ks
#include "cuda_randomFuncs.h"//setup_kernel,ran_gamma
#include "cuda_gillespie.h"//sim_gillespie

int main(int argc,char* argv[]){
  double increment = 1.0/(double)num_cells;

  double** raw_data;
  raw_data = (double**)malloc(MAX_RNA * sizeof(double*));
  for (int i=0;i<MAX_RNA;i++){
    raw_data[i] = (double*)malloc(MAX_RNA * sizeof(double));
  }
  read_data(argv[1],raw_data);
  double** raw_CDF1;
  raw_CDF1 = (double**)malloc(MAX_RNA * sizeof(double*));
  for (int i=0;i<MAX_RNA;i++){
    raw_CDF1[i] = (double*)malloc(MAX_RNA * sizeof(double));
  }
  double** raw_CDF2;
  raw_CDF2 = (double**)malloc(MAX_RNA * sizeof(double*));
  for (int i=0;i<MAX_RNA;i++){
    raw_CDF2[i] = (double*)malloc(MAX_RNA * sizeof(double));
  }
  double** raw_CDF3;
  raw_CDF3 = (double**)malloc(MAX_RNA * sizeof(double*));
  for (int i=0;i<MAX_RNA;i++){
    raw_CDF3[i] = (double*)malloc(MAX_RNA * sizeof(double));
  }
  double** raw_CDF4;
  raw_CDF4 = (double**)malloc(MAX_RNA * sizeof(double*));
  for (int i=0;i<MAX_RNA;i++){
    raw_CDF4[i] = (double*)malloc(MAX_RNA * sizeof(double));
  }
  calculate_CDF(raw_data,raw_CDF1,raw_CDF2,raw_CDF3,raw_CDF4);
  free(raw_data);
  //Create vectors with parameters chosen from a latin-hypercube
  double* max_param;
  max_param = (double*)malloc(6 * sizeof(double));//Set maximum values of each parameter
  max_param[0] = 2.0;
  max_param[1] = 2.0;
  max_param[2] = 4.0;
  max_param[3] = 5.0;
  max_param[4] = 5.0;
  max_param[5] = 0.5;
  double** chosen_params;
  chosen_params = (double**)malloc(6 * sizeof(double*));
  for (int i=0;i<6;i++){
    chosen_params[i] = (double*)malloc(num_params * sizeof(double));
  }
  create_paramsVecs(max_param,chosen_params);
  free(max_param);

  //Simulate distributions, calculate CDFs and KS statistics
  double** rna_dist;
  rna_dist = (double**)malloc(MAX_RNA * sizeof(double*));
  for (int i=0;i<MAX_RNA;i++){
    rna_dist[i] = (double*)malloc(MAX_RNA * sizeof(double));
  }

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
  fprintf(fp,"ON\tOFF\tINIT\tELONG\tEXP\tDEG\tKS\n");

  int* recv_nuc;
  recv_nuc = (int*)malloc(num_cells * sizeof(int));
  int* recv_cyt;
  recv_cyt = (int*)malloc(num_cells * sizeof(int));
  int* device_nuc;
  cudaMalloc((void**)&device_nuc,num_cells * sizeof(int));
  int* device_cyt;
  cudaMalloc((void**)&device_cyt,num_cells * sizeof(int));

  double* params;
  cudaMalloc((void**)&params,6 * sizeof(double));
  double* it_params;
  it_params = (double*)malloc(6 * sizeof(double));

  for (int iteration = 0;iteration<num_params;iteration++){
    for (int i=0;i<MAX_RNA;i++){
      for (int j=0;j<MAX_RNA;j++){
        rna_dist[i][j] = 0.0;
      }
    }

    for (int i=0;i<6;i++){it_params[i] = chosen_params[i][iteration];}
    cudaMemcpy(params,it_params,6 * sizeof(double),cudaMemcpyHostToDevice);

    sim_gillespie<<<num_blocks,num_threads>>>(params,device_nuc,device_cyt,devStates);

    cudaMemcpy(recv_nuc,device_nuc,num_cells * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(recv_cyt,device_cyt,num_cells * sizeof(int),cudaMemcpyDeviceToHost);

    for (int cell=0;cell<num_cells;cell++){
      if ((recv_nuc[cell]<MAX_RNA)&&(recv_cyt[cell]<MAX_RNA)){
        rna_dist[recv_nuc[cell]][recv_cyt[cell]]+=increment;
      }
    }
    double ks_stat;
    ks_stat = calculate_ks(raw_CDF1,raw_CDF2,raw_CDF3,raw_CDF4,rna_dist);
    for (int i=0;i<6;i++){
      fprintf(fp,"%f\t",it_params[i]);
    }
    fprintf(fp,"%f\n",ks_stat);
  }
  fclose(fp);

  cudaFree(device_nuc);
  cudaFree(device_cyt);
  cudaFree(params);
  cudaFree(devStates);

  free(recv_nuc);
  free(recv_cyt);
  free(it_params);

  free(raw_CDF1);
  free(raw_CDF2);
  free(raw_CDF3);
  free(raw_CDF4);
  free(chosen_params);

  return 0;
}
