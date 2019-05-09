#include "rnn.h"

/* C++ code here. */

void rnnConfig::setTopology1(Topology *top){
  cTopology1 = top;
}

void rnnConfig::setTopology2(Topology *top){
  cTopology2 = top;
}

void rnnConfig::setTopology3(Topology *top){
  cTopology3 = top;
}

void rnnConfig::setTopology4(Topology *top){
  cTopology4 = top;
}

void rnnConfig::setM(int M){
  mM=M;
}

Topology* rnnConfig::getTopology1(){
  return cTopology1;
}

Topology* rnnConfig::getTopology2(){
  return cTopology2;
}

Topology* rnnConfig::getTopology3(){
  return cTopology3;
}

Topology* rnnConfig::getTopology4(){
  return cTopology4;
}

int rnnConfig::getM(){
  return mM;
}

//***********************************
//
//RnnSerialDBL
//
void RnnSerialDBL::prepare(rnnConfig *mRnnConf){
  M = mRnnConf->getM();

  ann1 = new AnnSerialDBL(mRnnConf->getTopology1(),M);
  ann2 = new AnnSerialDBL(mRnnConf->getTopology2(),M);
  ann3 = new AnnSerialDBL_tanh(mRnnConf->getTopology3(),M);
  ann4 = new AnnSerialDBL(mRnnConf->getTopology4(),M);

  b = new double[M];
  c_current = new double[M];
  c_new = new double[M];
  h_current = new double[M];
  h_new = new double[M];
  a1_output = new double[M];
  a2_output = new double[M];
  a3_output = new double[M];
  a4_output = new double[M];

	// a_arr = new double[neuronCount];
	// z_arr = new double[neuronCount];
  //
  // ah_arr = new double[mM];
  //
	// W = new int[cTopology->getLayerCount()];
	// sw = new int[cTopology->getLayerCount()];
  //
	// w_arr = new double[weightCount];
	// dw_arr = new double[weightCount];
  //
  // int nd_layer_size = cTopology->getLayerSize(2);
  // int h_weightCount = nd_layer_size * M;
}


void RnnSerialDBL::init(FILE * pFile=NULL){

  for(int i=0; i<M;i++){
    c_current[i] = 0;
    h_current[i] = 0;
  }

}

void RnnSerialDBL::feedForward(double *h_in, double *c_in,double *a, double *c_out, double *h_out){
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


AnnSerialDBL* RnnSerialDBL::getANN1(){
  return ann1;
}

AnnSerialDBL* RnnSerialDBL::getANN2(){
  return ann2;
}

AnnSerialDBL_tanh* RnnSerialDBL::getANN3(){
  return ann3;
}

AnnSerialDBL* RnnSerialDBL::getANN4(){
  return ann4;
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
