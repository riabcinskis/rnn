#include "rnn.h"

/* C++ code here. */
void rnnConfig::setTopology(Topology **top){
  cTopology = top;
}

void rnnConfig::setTopology1(Topology *top){
  cTopology[0] = top;
}

void rnnConfig::setTopology2(Topology *top){
  cTopology[1] = top;
}

void rnnConfig::setTopology3(Topology *top){
  cTopology[2] = top;
}

void rnnConfig::setTopology4(Topology *top){
  cTopology[3] = top;
}

void rnnConfig::setM(int M){
  mM=M;
}

Topology* rnnConfig::getTopology1(){
  return cTopology[0];
}

Topology* rnnConfig::getTopology2(){
  return cTopology[1];
}

Topology* rnnConfig::getTopology3(){
  return cTopology[2];
}

Topology* rnnConfig::getTopology4(){
  return cTopology[3];
}

Topology** rnnConfig::getTopology(){
  return cTopology;
}

int rnnConfig::getM(){
  return mM;
}

//***********************************
//
//RnnSerialDBL
//
void RnnSerial::prepare(rnnConfig *mRnnConf){
  M = mRnnConf->getM();

  double (*func)(double);
  double (*func_deriv)(double);

  double (*func_tanh)(double);
  double (*func_tanh_deriv)(double);

  func=f;
  func_deriv = f_deriv;

  func_tanh=f_tanh;
  func_tanh_deriv = f_tanh_deriv;


  ann1 = new AnnSerial(4,0,M,mRnnConf->getTopology(),func,func_deriv);
  ann2 = new AnnSerial(4,1,M,mRnnConf->getTopology(),func,func_deriv);
  ann3 = new AnnSerial(4,2,M,mRnnConf->getTopology(),func_tanh,func_tanh_deriv);
  ann4 = new AnnSerial(4,3,M,mRnnConf->getTopology(),func,func_deriv);



  b = new double[M];
  c_current = new double[M];
  c_new = new double[M];
  h_current = new double[M];
  h_new = new double[M];
  a1_output = new double[M];
  a2_output = new double[M];
  a3_output = new double[M];
  a4_output = new double[M];

}


void RnnSerial::init(FILE * pFile=NULL){

  for(int i=0; i<M;i++){
    c_current[i] = 0;
    h_current[i] = 0;
  }

}

void RnnSerial::feedForward(double *h_in, double *c_in,double *a, double *c_out, double *h_out){
  ann1->feedForward(h_in,a,a1_output);

  ann2->feedForward(h_in,a,a2_output);

  ann3->feedForward(h_in,a,a3_output);

  ann4->feedForward(h_in,a,a4_output);

  double temp= 0;

  for(int i=0; i < M; i++){
    temp = c_in[i] * a1_output[i] + a2_output[i] * a3_output[i];
    c_out[i] = temp;
    temp = tanh(temp);
    b[i] = temp * a4_output[i];
  }

  double sumB = 0;
  for(int i = 0; i < M; i++){
    sumB+= b[i];
  }

  for(int i = 0; i < M; i++){
    h_out[i] = b[i] / sumB;
  }
}


void RnnSerial::backPropagation(){

}

void RnnSerial::destroy(){
  ann1->destroy();
  ann2->destroy();
  ann3->destroy();
  ann4->destroy();

   delete ann1;
   ann1 = NULL;
   delete ann2;
   ann2 = NULL;
   delete ann3;
   ann3 = NULL;
   delete ann4;
   ann4 = NULL;

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
  delete a1_output;
  a1_output = NULL;
  delete a2_output;
  a2_output = NULL;
  delete a3_output;
  a3_output = NULL;
  delete a4_output;
  a4_output = NULL;
}


AnnSerial* RnnSerial::getANN1(){
  return ann1;
}

AnnSerial* RnnSerial::getANN2(){
  return ann2;
}

AnnSerial* RnnSerial::getANN3(){
  return ann3;
}

AnnSerial* RnnSerial::getANN4(){
  return ann4;
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

// double* RnnSerialDBL::getWeights(){
// 	return w_arr;
// }
//
// double* RnnSerialDBL::getDWeights(){
// 	return dw_arr;
// }
//
// double* RnnSerialDBL::getHWeights(){
// 	return wh_arr;
// }
//
// double* RnnSerialDBL::getDHWeights(){
// 	return dwh_arr;
// }
//
// double* RnnSerialDBL::getA(){
// 	return a_arr;
// }
//
// Topology* RnnSerialDBL::getTopology(){
//   return cTopology;
// }
