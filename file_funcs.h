void read_data(char* filename,double** raw_data){
  //Read data from file
  FILE *fp;
  fp = fopen(filename,"r");
  char buffer[1024];
  const char s[2] = "\t";
  char *token;
  int i=0;
  while (fgets(buffer,sizeof(buffer),fp)){
    token = strtok(buffer,s);
    for (int j=0;j<MAX_RNA;j++){
      raw_data[i][j] = strtod(token,NULL);
      token = strtok(NULL,s);
    }
    i++;
  }
  fclose(fp);

  //Normalise data
  /*
  double tot_rna = 0.0;
  for (int i=0;i<MAX_RNA;i++){
    for (int j=0;j<MAX_RNA;j++){
      tot_rna += raw_data[i][j];
    }
  }
  for (int i=0;i<MAX_RNA;i++){
    for (int j=0;j<MAX_RNA;j++){
      raw_data[i][j] /= tot_rna;
      printf("%f ",raw_data[i][j]);
    }
    printf("\n");
  }
  */
}

void read_1DData(char* filename,double* data_vec){
  //Read data from file
  FILE *fp;
  fp = fopen(filename,"r");
  char buffer[256];
  int i=0;
  while (fgets(buffer,sizeof(buffer),fp)){
    data_vec[i] = strtod(buffer,NULL);
    i++;
  }
  fclose(fp);

  //Normalise data
  /*
  double tot_rna = 0.0;
  for (int i=0;i<MAX_RNA;i++){
    tot_rna += data_vec[i];
  }
  for (int i=0;i<MAX_RNA;i++){
    data_vec[i] /= tot_rna;
  }
  */
}
