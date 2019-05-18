#include "rnn.h"

/* C++ code here. */
void RnnConfig::setTopologies(Topology **top){
  cTopology = top;
}

void RnnConfig::setM(int M){
  mM=M;
}


Topology** RnnConfig::getTopologies(){
  return cTopology;
}

int RnnConfig::getM(){
  return mM;
}


//
// RnnSerialDBL
//
void RnnSerial::prepare(int M, Topology **top){
  this->M = M;
  V = 4;

  double (*func)(double);
  double (*func_deriv)(double);

  double (*func_tanh)(double);
  double (*func_tanh_deriv)(double);

  func=f;
  func_deriv = f_deriv;

  func_tanh=f_tanh;
  func_tanh_deriv = f_tanh_deriv;


  anns = new AnnSerial*[V];
  for(int v = 0; v < V; v++){
    if(v==2){
      anns[v] = new AnnSerial(V, v, M, top, func_tanh, func_tanh_deriv);
    } else {
      anns[v] = new AnnSerial(V, v, M, top, func, func_deriv);
    }
  }

  ann_forget = anns[0];
  ann_input = anns[1];
  ann_gate = anns[2];
  ann_output = anns[3];




  b = new double[M];
  c_current = new double[M];
  c_new = new double[M];
  h_current = new double[M];
  h_new = new double[M];

  a_outputs = new double*[V];
  for(int v = 0; v < V; v++){
    a_outputs[v] = new double[M];
  }
}


void RnnSerial::init(FILE * pFile=NULL){

  for(int i=0; i<M; i++){
    c_current[i] = 0;
    h_current[i] = 0;
  }

}

void RnnSerial::feedForward(double *h_in, double *c_in, double *a_in, double *c_out, double *h_out){

  for(int v = 0; v < V; v++)
    anns[v]->feedForward(h_in, a_in, a_outputs[v]);

  for(int i=0; i < M; i++){
    c_out[i] = c_in[i] * a_outputs[0][i] + a_outputs[1][i] * a_outputs[2][i];
    b[i] = tanh(c_out[i])* a_outputs[3][i];
  }

  double sumB = 0;
  for(int i = 0; i < M; i++)
    sumB += exp(b[i]);

  for(int i = 0; i < M; i++)
    h_out[i] = exp(b[i]) / sumB;

}


void RnnSerial::backPropagation(Derivatives **deriv_in, Derivatives **deriv_out){

  for(int v = 0; v < V; v++)
    anns[v]->backPropagation(deriv_in, deriv_out);//[v]./



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
