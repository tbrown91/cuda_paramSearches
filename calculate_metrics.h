void calculate_CDF(double** raw_data,double** CDF1,double** CDF2, double** CDF3, double** CDF4){
  //Calculate the 2D CDF to be sent to the kernels
  //First CDF calculated via {X<x,Y<y}
  CDF1[0][0]=raw_data[0][0];//First row of cdf matrix
  for (int j=1;j<MAX_RNA;++j){
    CDF1[0][j] = CDF1[0][j-1]+raw_data[0][j];//Fill in first row adding along the row
  }
  for (int i=1;i<MAX_RNA;++i){
    CDF1[i][0] = (CDF1[i-1][0]+raw_data[i][0]);//First element of row is cdf from above plus its own pdf
    for (int j=1;j<MAX_RNA;++j){
      CDF1[i][j] = (CDF1[i][j-1]+raw_data[i][j]);//Sum of pdfs from columns to the left
      for (int k=0;k<i;++k){
        CDF1[i][j]+=raw_data[k][j];//Sum of pdfs from current column
      }
    }
  }

  //Second CDF calculated via {X<x,Y>y}
  for (int i=0;i<MAX_RNA;++i){
    CDF2[i][0]=(raw_data[i][0]);
    for (int k=(i+1);k<MAX_RNA;++k){
      CDF2[i][0]+=raw_data[k][0];
    }
    for (int j=1;j<MAX_RNA;++j){
      CDF2[i][j]=(CDF2[i][j-1]);
      for (int k=i;k<MAX_RNA;++k){
        CDF2[i][j]+=raw_data[k][j];
      }
    }
  }

  //Third CDF calculated through criteria F(x,y) = P(X>x,Y<y)
  CDF3[0][0]=(raw_data[0][0]);
  for (int j=1;j<MAX_RNA;++j){
    CDF3[0][0]+=raw_data[0][j];
  }
  for (int j=1;j<MAX_RNA;++j){
    CDF3[0][j]=(raw_data[0][j]);
    for (int k=(j+1);k<MAX_RNA;++k){
      CDF3[0][j]+=raw_data[0][k];
    }
  }
  for (int i=1;i<MAX_RNA;++i){
    CDF3[i][0]=(CDF3[i-1][0]);
    for (int j=0;j<MAX_RNA;++j){
      CDF3[i][0]+=raw_data[i][j];
    }
    for (int j=1;j<MAX_RNA;++j){
      CDF3[i][j]=(CDF3[i-1][j]);
      for (int k=j;k<MAX_RNA;++k){
        CDF3[i][j]+=raw_data[i][k];
      }
    }
  }

  //Fourth CDF calculated through criteria F(x,y) = P(X>x,Y>y)
  CDF4[MAX_RNA-1][MAX_RNA-1] = raw_data[MAX_RNA-1][MAX_RNA-1];
  for (int j=MAX_RNA-2;j>=0;--j){
    CDF4[MAX_RNA-1][j] = CDF4[MAX_RNA-1][j+1] + raw_data[MAX_RNA-1][j];
  }
  for (int i=MAX_RNA-2;i>=0;--i){
    CDF4[i][MAX_RNA-1] = CDF4[i+1][MAX_RNA-1] + raw_data[i][MAX_RNA-1];
    for (int j=MAX_RNA-2;j>=0;--j){
      CDF4[i][j] = CDF4[i+1][j];
      for (int k=j;k<MAX_RNA;++k){
        CDF4[i][j]+=raw_data[i][k];
      }
    }
  }

}

double calculate_ks(double** raw_CDF1,double** raw_CDF2,double** raw_CDF3,
  double** raw_CDF4,double** rna_dist){

  double CDF1[MAX_RNA][MAX_RNA];
  double CDF2[MAX_RNA][MAX_RNA];
  double CDF3[MAX_RNA][MAX_RNA];
  double CDF4[MAX_RNA][MAX_RNA];

  //First CDF calculated via {X<x,Y<y}
  CDF1[0][0]=rna_dist[0][0];//First row of cdf matrix
  for (int j=1;j<MAX_RNA;++j){
    CDF1[0][j] = CDF1[0][j-1]+rna_dist[0][j];//Fill in first row adding along the row
  }
  for (int i=1;i<MAX_RNA;++i){
    CDF1[i][0] = (CDF1[i-1][0]+rna_dist[i][0]);//First element of row is cdf from above plus its own pdf
    for (int j=1;j<MAX_RNA;++j){
      CDF1[i][j] = (CDF1[i][j-1]+rna_dist[i][j]);//Sum of pdfs from columns to the left
      for (int k=0;k<i;++k){
        CDF1[i][j]+=rna_dist[k][j];//Sum of pdfs from current column
      }
    }
  }

  //Second CDF calculated via {X<x,Y>y}
  for (int i=0;i<MAX_RNA;++i){
    CDF2[i][0]=(rna_dist[i][0]);
    for (int k=(i+1);k<MAX_RNA;++k){
      CDF2[i][0]+=rna_dist[k][0];
    }
    for (int j=1;j<MAX_RNA;++j){
      CDF2[i][j]=(CDF2[i][j-1]);
      for (int k=i;k<MAX_RNA;++k){
        CDF2[i][j]+=rna_dist[k][j];
      }
    }
  }

  //Third CDF calculated through criteria F(x,y) = P(X>x,Y<y)
  CDF3[0][0]=(rna_dist[0][0]);
  for (int j=1;j<MAX_RNA;++j){
    CDF3[0][0]+=rna_dist[0][j];
  }
  for (int j=1;j<MAX_RNA;++j){
    CDF3[0][j]=(rna_dist[0][j]);
    for (int k=(j+1);k<MAX_RNA;++k){
      CDF3[0][j]+=rna_dist[0][k];
    }
  }
  for (int i=1;i<MAX_RNA;++i){
    CDF3[i][0]=(CDF3[i-1][0]);
    for (int j=0;j<MAX_RNA;++j){
      CDF3[i][0]+=rna_dist[i][j];
    }
    for (int j=1;j<MAX_RNA;++j){
      CDF3[i][j]=(CDF3[i-1][j]);
      for (int k=j;k<MAX_RNA;++k){
        CDF3[i][j]+=rna_dist[i][k];
      }
    }
  }

  //Fourth CDF calculated through criteria F(x,y) = P(X>x,Y>y)
  CDF4[MAX_RNA-1][MAX_RNA-1] = rna_dist[MAX_RNA-1][MAX_RNA-1];
  for (int j=MAX_RNA-2;j>=0;--j){
    CDF4[MAX_RNA-1][j] = CDF4[MAX_RNA-1][j+1] + rna_dist[MAX_RNA-1][j];
  }
  for (int i=MAX_RNA-2;i>=0;--i){
    CDF4[i][MAX_RNA-1] = CDF4[i+1][MAX_RNA-1] + rna_dist[i][MAX_RNA-1];
    for (int j=MAX_RNA-2;j>=0;--j){
      CDF4[i][j] = CDF4[i+1][j];
      for (int k=j;k<MAX_RNA;++k){
        CDF4[i][j]+=rna_dist[i][k];
      }
    }
  }

  //Now calculate KS statistic
  double ks_stat = 0.0;
  for (int i=0;i<MAX_RNA;i++){
    for (int j=0;j<MAX_RNA;j++){
      ks_stat = MAX(ks_stat,fabs(CDF1[i][j] - raw_CDF1[i][j]));
      ks_stat = MAX(ks_stat,fabs(CDF2[i][j] - raw_CDF2[i][j]));
      ks_stat = MAX(ks_stat,fabs(CDF3[i][j] - raw_CDF3[i][j]));
      ks_stat = MAX(ks_stat,fabs(CDF4[i][j] - raw_CDF4[i][j]));
    }
  }

  return ks_stat;
}

void calculate_1DCDF(double* dist,double* CDF){
  CDF[0] = dist[0];
  for (int i=1;i<MAX_RNA;i++){
    CDF[i] = CDF[i-1] + dist[i];
  }
}

double calculate_1DKS(double* raw_CDF,double* sim_dist,double* sim_CDF){
  sim_CDF[0] = sim_dist[0];
  double ks_stat = fabs(raw_CDF[0] - sim_CDF[0]);
  for (int i=1;i<MAX_RNA;i++){
    sim_CDF[i] = sim_CDF[i-1] + sim_dist[i];
    ks_stat = MAX(ks_stat,fabs(raw_CDF[i] - sim_CDF[i]));
  }
  return ks_stat;
}
