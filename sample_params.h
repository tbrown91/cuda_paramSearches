void choose_params(double* possible_params,double* chosen_params){
  for (int param = 0;param < num_params;param++){
    int param_index = floor((rand()/((double)(RAND_MAX)+1.0))*(num_params-param));
    chosen_params[param] = possible_params[param_index];
    possible_params[param_index] = possible_params[num_params - 1 - param];
  }
}

void create_paramsVecs(double* max_param,double** chosen_params){
  double** possible_params;
  possible_params = (double**)malloc(6 * sizeof(double*));
  for (int i=0;i<6;i++){
    possible_params[i] = (double*)malloc(num_params * sizeof(double));
  }
  for (int i=0;i<6;++i){
    for (int j=1;j<=num_params;++j){
      possible_params[i][j-1] = (max_param[i]/(double)num_params)*(double)j;
    }
  }
  for (int i=0;i<6;i++){
    choose_params(possible_params[i],chosen_params[i]);
  }
  free(possible_params);
}

void create_paramsVecsReduced(double* max_param,double** chosen_params){
  double** possible_params;
  possible_params = (double**)malloc(5 * sizeof(double*));
  for (int i=0;i<5;i++){
    possible_params[i] = (double*)malloc(num_params * sizeof(double));
  }
  for (int i=0;i<5;++i){
    for (int j=1;j<=num_params;++j){
      possible_params[i][j-1] = (max_param[i]/(double)num_params)*(double)j;
    }
  }
  for (int i=0;i<4;i++){
    choose_params(possible_params[i],chosen_params[i]);
  }
  for (int j=0;j<num_params;++j){
    chosen_params[4][j]=0.015;//Fix degradation rate
  }
  free(possible_params);
}

void create_paramsVecsNoBurst(double* max_param,double** chosen_params){
  double** possible_params;
  possible_params = (double**)malloc(3 * sizeof(double*));
  for (int i=0;i<3;i++){
    possible_params[i] = (double*)malloc(num_params * sizeof(double));
  }
  for (int i=0;i<3;++i){
    for (int j=1;j<=num_params;++j){
      possible_params[i][j-1] = (max_param[i]/(double)num_params)*(double)j;
    }
  }
  for (int i=0;i<2;i++){
    choose_params(possible_params[i],chosen_params[i]);
  }
  for (int j=0;j<num_params;++j){
    chosen_params[2][j]=0.015;//Fix degradation rate
  }
  free(possible_params);
}
