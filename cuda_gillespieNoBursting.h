__global__ void sim_gillespie(double* params,int* device_nuc,int* device_cyt,curandState *states){

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < num_cells){
    curandState localState = states[tid];
    //Simulate a single cell and update system
    int RNA = 0;
    int num_starts = 0;
    int num_finishes = 0;
    int finish_times[1000];
    double current_time = 0.0;
    double propensities[2];
    double elong_rate = 1.0/(params[1]*1000.0);
    while (current_time < 500.0){
      propensities[0] = params[0];//Initiation
      propensities[1] = (double)RNA*params[2];//Degradation

      double tot_prop = 0;
      for (int i=0;i<2;i++){
        tot_prop += propensities[i];
      }

      current_time -= (1.0/tot_prop)*log(1 - curand_uniform_double(&localState));

      if (num_starts > num_finishes){
        if (current_time > finish_times[num_finishes%1000]){
          current_time = finish_times[num_finishes%1000];
          RNA++;
          num_finishes++;
          continue;
        }
      }

      double reac_rand = curand_uniform_double(&localState)*tot_prop;
      int reac_index = 0;
      while (reac_rand > propensities[reac_index]){
        reac_rand -= propensities[reac_index];
        reac_index++;
      }

      switch (reac_index) {
        case 0:
        finish_times[num_starts%1000] = current_time + ran_gamma(localState,gene_length,elong_rate);
        num_starts++;
        break;
        case 1:
        RNA--;
        break;
      }

    }
    device_nuc[tid] = (num_starts - num_finishes);
    device_cyt[tid] = RNA;

    states[tid] = localState;
    tid += blockDim.x * gridDim.x;
  }

}
