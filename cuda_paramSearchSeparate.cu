#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda.h"
#include "curand.h"
#include "curand_kernel.h"

const int num_params = 1000000;
const int MAX_RNA = 50;
const int num_cells = 10000;
const int gene_length = 1587;//GAL1 1587, GAL10 2100

double MAX(double a,double b){
  if (a > b){return a;}
  else{return b;}
}

#include "file_funcs.h"//read_1DData,write_resultsSeparate
#include "sample_params.h"//choose_params,create_paramsVecs
#include "calculate_metrics.h"//calculate_1DCDF,calculate_1DKS
#include "cuda_randomFuncs.h"//setup_kernel,ran_gamma
#include "cuda_gillespie.h"//sim_gillespie

int main(int argc,char* argv[]){
  double increment = 1.0/(double)num_cells;

  double* raw_nuc;
  raw_nuc = (double*)malloc(MAX_RNA * sizeof(double));
  read_1DData(argv[1],raw_nuc);
  double* raw_nucCDF;
  raw_nucCDF = (double*)malloc(MAX_RNA * sizeof(double));
  double* raw_cyt;
  raw_cyt = (double*)malloc(MAX_RNA * sizeof(double));
  read_1DData(argv[2],raw_cyt);
  double* raw_cytCDF;
  raw_cytCDF = (double*)malloc(MAX_RNA * sizeof(double));

  calculate_1DCDF(raw_nuc,raw_nucCDF);
  calculate_1DCDF(raw_cyt,raw_cytCDF);
  free(raw_nuc);
  free(raw_cyt);
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

  double* nuc_distCDF;
  nuc_distCDF = (double*)malloc(MAX_RNA * sizeof(double));
  double* cyt_distCDF;
  cyt_distCDF = (double*)malloc(MAX_RNA * sizeof(double));

  FILE *fp;
  fp = fopen(argv[3],"w");
  fprintf(fp,"ON\tOFF\tINIT\tELONG\tEXP\tDEG\tKS_NUC\tKS_CYT\n");

  double ks_statNuc,ks_statCyt;

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
      nuc_dist[i] = 0.0;
      cyt_dist[i] = 0.0;
    }
    for (int i=0;i<6;i++){it_params[i] = chosen_params[i][iteration];}
    cudaMemcpy(params,it_params,6 * sizeof(double),cudaMemcpyHostToDevice);

    sim_gillespie<<<num_blocks,num_threads>>>(params,device_nuc,device_cyt,devStates);

    cudaMemcpy(recv_nuc,device_nuc,num_cells * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(recv_cyt,device_cyt,num_cells * sizeof(int),cudaMemcpyDeviceToHost);

    for (int cell=0;cell<num_cells;cell++){
      if ((recv_nuc[cell]<MAX_RNA)&&(recv_cyt[cell]<MAX_RNA)){
        nuc_dist[recv_nuc[cell]] += increment;
        cyt_dist[recv_cyt[cell]] += increment;
      }
    }

    ks_statNuc = calculate_1DKS(raw_nucCDF,nuc_dist,nuc_distCDF);
    ks_statCyt = calculate_1DKS(raw_cytCDF,cyt_dist,cyt_distCDF);
    for (int i=0;i<6;i++){
      fprintf(fp,"%f\t",it_params[i]);
    }
    fprintf(fp,"%f\t%f\n",ks_statNuc,ks_statCyt);
  }
  fclose (fp);

  cudaFree(device_nuc);
  cudaFree(device_cyt);
  cudaFree(params);

  free(it_params);
  free(recv_nuc);
  free(recv_cyt);
  free(nuc_dist);
  free(cyt_dist);
  free(nuc_distCDF);
  free(cyt_distCDF);
  free(raw_nucCDF);
  free(raw_cytCDF);

  free(chosen_params);

  return 0;
}
