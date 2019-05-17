#include "rnn.h"

/* C++ code here. */
void rnnConfig::setTopology(Topology **top){
  cTopology = top;
}

void rnnConfig::setM(int M){
  mM=M;
}

void rnnConfig::setV(int V){
  mV=V;
}

Topology** rnnConfig::getTopology(){
  return cTopology;
}

int rnnConfig::getM(){
  return mM;
}

int rnnConfig::getV(){
  return mV;
}

//***********************************
//
//RnnSerialDBL
//
void RnnSerial::prepare(rnnConfig *mRnnConf){
  M = mRnnConf->getM();
  V = mRnnConf->getV();

  double (*func)(double);
  double (*func_deriv)(double);

  double (*func_tanh)(double);
  double (*func_tanh_deriv)(double);

  func=f;
  func_deriv = f_deriv;

  func_tanh=f_tanh;
  func_tanh_deriv = f_tanh_deriv;


  anns = new AnnSerial*[V];
  for(int i = 0; i < V; i++){
    if(i==2){
      anns[i] = new AnnSerial(V,i,M,mRnnConf->getTopology(),func_tanh,func_tanh_deriv);
    } else {
      anns[i] = new AnnSerial(V,i,M,mRnnConf->getTopology(),func,func_deriv);
    }
  }

  b = new double[M];
  c_current = new double[M];
  c_new = new double[M];
  h_current = new double[M];
  h_new = new double[M];

  a_outputs = new double*[V];
  for(int i = 0; i < V; i++){
    a_outputs[i] = new double[M];
  }
}


void RnnSerial::init(FILE * pFile=NULL){

  for(int i=0; i<M;i++){
    c_current[i] = 0;
    h_current[i] = 0;
  }

}

void RnnSerial::feedForward(double *h_in, double *c_in,double *a, double *c_out, double *h_out){

  for(int i = 0; i < V; i++)
    anns[i]->feedForward(h_in,a,a_outputs[i]);

  double temp= 0;

  // for(int i=0; i < M; i++){
  //   temp = c_in[i] * a1_output[i] + a2_output[i] * a3_output[i];
  //   c_out[i] = temp;
  //   temp = tanh(temp);
  //   b[i] = temp * a4_output[i];
  // }

  // double sumB = 0;
  // for(int i = 0; i < M; i++){
  //   sumB+= b[i];
  // }
  //
  // for(int i = 0; i < M; i++){
  //   h_out[i] = b[i] / sumB;
  // }
}


void RnnSerial::backPropagation(double *h_in, double *a, Derivatives **deriv_in, Derivatives **deriv_out){
  for(int i = 0; i < V; i++)
    anns[i]->feedForward(h_in,a,a_outputs[i]);


  double *aoutput = new double[M];

  for(int i = 0; i < V; i++)
    anns[i]->backPropagation(deriv_in, deriv_out, aoutput);



    printf("%.20f\n", deriv_out[0]->v[anns[0]->vi(0,0,0,0,0)]);


}

void RnnSerial::destroy(){
    for(int i = 0; i < V; i++){
      anns[i]->destroy();
    }

   delete [] anns;
   anns = NULL;


  delete c_current;
  c_current = NULL;
  delete c_new;
  c_new = NULL;
  delete h_current;
  h_current = NULL;
  delete h_new;
  h_new = NULL;
  delete b;
  b = NULL;
  delete [] a_outputs;
  a_outputs = NULL;

}


AnnSerial* RnnSerial::getANN(int v){
  return anns[v];
}


double f(double x){
  double y = 1 + exp(-x);
  return 1 / y;
}

double f_deriv(double x){
  return exp(-x) / pow((1 + exp(-x)), 2);
}

double f_tanh(double x){
  return tanh(x);
}

double f_tanh_deriv(double x){
  return 1 - pow(tanh(x), 2);
}
