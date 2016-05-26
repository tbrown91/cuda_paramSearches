__global__ void sim_gillespie(double* params,int* device_nuc,int* device_cyt,curandState *states){

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < num_cells){
    curandState localState = states[tid];
    //Simulate a single cell and update system
    int promoter = 0;
    int RNA[2] = {0.0};
    int num_starts = 0;
    int num_finishes = 0;
    int finish_times[1000];
    double current_time = 0.0;
    double propensities[5];
    double elong_rate = 1.0/(params[3]*1000.0);
    while (current_time < 500.0){
      propensities[0] = (double)(1-promoter)*params[0];
      propensities[1] = (double)(promoter)*params[1];
      propensities[2] = (double)(promoter)*params[2];
      propensities[3] = (double)RNA[0]*params[4];
      propensities[4] = (double)RNA[1]*params[5];

      double tot_prop = 0;
      for (int i=0;i<5;i++){
        tot_prop += propensities[i];
      }

      current_time -= (1.0/tot_prop)*log(1 - curand_uniform_double(&localState));

      if (num_starts > num_finishes){
        if (current_time > finish_times[num_finishes%1000]){
          current_time = finish_times[num_finishes%1000];
          RNA[0]++;
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
        promoter = 1;
        break;
        case 1:
        promoter = 0;
        break;
        case 2:
        finish_times[num_starts%1000] = current_time + ran_gamma(localState,gene_length,elong_rate);
        num_starts++;
        break;
        case 3:
        RNA[0]--;
        RNA[1]++;
        break;
        case 4:
        RNA[1]--;
        break;
      }

    }
    device_nuc[tid] = (num_starts - num_finishes + RNA[0]);
    device_cyt[tid] = RNA[1];

    states[tid] = localState;
    tid += blockDim.x * gridDim.x;
  }

}
