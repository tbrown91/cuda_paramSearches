__global__ void setup_kernel(curandState *states){//Use this function to seed random numbers
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int seed = (int) clock64();
  /* Each thread gets same seed, a different sequence
       number, no offset */
  while (id < num_cells){
    curand_init(seed, id, 0, &states[id]);
    id += blockDim.x * gridDim.x;
  }
}


__device__ double ran_gamma (curandState localState, const double a, const double b){
  double x, v, u;
  double d = a - 1.0 / 3.0;
  double c = (1.0 / 3.0) / sqrt (d);

  while (1){
      do{
          x = curand_normal_double(&localState);
          v = 1.0 + c * x;
      } while (v <= 0);

      v = v * v * v;
      u = curand_uniform_double(&localState);

      if (u < 1 - 0.0331 * x * x * x * x)
          break;

      if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
          break;
  }
  return b * d * v;
}
