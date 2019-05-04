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




}


// void RnnSerialDBL::feedForward(double *h_input,double *a, double *b){
// 	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
// 		a_arr[i] = a[i];
// 	}
//
//   for(int i=0; i<M;i++){
//     ah_arr[i] = h_input[i];
//   }
//
// 	for (int j = 0; j < cTopology->obtainNeuronCount(); j++) {
// 		z_arr[j] = 0;
// 	}
//
// 	calc_feedForward();
//
// 	for (int i = 0; i<cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++)
// 		b[i] = a_arr[s[L - 1] + i];
// }
//
// void RnnSerialDBL::calc_feedForward(){
//   for(int i = 0; i < l[1] - 1; i++){
//     for(int j = 0; j < M; j++){
//       z_arr[s[1] + i] += ah_arr[j] * wh_arr[j*M + i];
//     }
//   }
// 	for (int i = 0; i < L - 1; i++) {//per sluoksnius einu+
// 		for (int j = 0; j < l[i]; j++) { //kiek neuronu sluoksnyje+
// 			for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z+
// 				z_arr[s[i + 1] + k] += w_arr[sw[i] + k + j*(l[i + 1] - 1)] * a_arr[s[i] + j];
// 			}
// 		}
// 		for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z
// 			a_arr[s[i + 1] + k] = f(z_arr[s[i + 1] + k]);
// 		}
// 	}
// }


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
