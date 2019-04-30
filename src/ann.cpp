#include "ann.h"

using namespace std;

//
// Random
//
Random::Random(){
  mGen = new std::mt19937();
  mDist = new std::uniform_real_distribution<double>(0., 1.);
}

double Random::next(){
  return (*mDist)(*mGen);
}

int Random::nextInt(int min, int max){
  double range = max - min;
  double r = range * next();
  return min + (int)(r+0.5);
}

bool Random::nextBool(){
  if(next() >= 0) return true;
  return false;
}

//*****************************
//
// Topology
//
Topology::Topology(){
	ml = new vector<int>();
}

Topology::~Topology(){
	ml->clear();
	delete ml;
}

void Topology::addLayer(int size){
	ml->push_back(size);
}

int Topology::getLayerCount(){
	return ml->size();
}

int Topology::getLayerSize(int index){
	return (*ml)[index];
}

int Topology::obtainNeuronCount(){
	int count = 0;
	for (int i = 0; i < ml->size(); i++)
		count += (*ml)[i] + 1;
	return count;
}

int Topology::obtainWeightCount(){
	int count = 0;
	for (int i = 0; i < ml->size()-1; i++)
		count += ((*ml)[i] + 1)*(*ml)[i+1];
	return count;
}

int Topology::getInputNeuronCount(){
	return (*ml)[0];
}

int Topology::getOutputNeuronCount(){
	return (*ml)[ml->size()-1];
}






//***********************************
//
//AnnSerialDBL
//
void AnnSerialDBL::prepare(Topology *top, int mM){

  cTopology = top;
  M = mM;

	l = new int[cTopology->getLayerCount()];
	s = new int[cTopology->getLayerCount()];

	int neuronCount = cTopology->obtainNeuronCount();
	int weightCount = cTopology->obtainWeightCount();

	a_arr = new double[neuronCount];
	z_arr = new double[neuronCount];

  ah_arr = new double[mM];

	W = new int[cTopology->getLayerCount()];
	sw = new int[cTopology->getLayerCount()];

	w_arr = new double[weightCount];
	dw_arr = new double[weightCount];

  int nd_layer_size = cTopology->getLayerSize(2);
  int h_weightCount = nd_layer_size * M;

  wh_arr = new double[h_weightCount];
  dwh_arr = new double[h_weightCount];

	//gjl = new double[neuronCount];
}


void AnnSerialDBL::init(FILE * pFile=NULL){
  L = cTopology->getLayerCount();

	Random *rnd = new Random();

	//Neuronu kiekiai sluoksnyje
	for (int i = 0; i < L; i++) {
		l[i] = cTopology->getLayerSize(i) + 1;
	}

	//Sluoksniu pradzios indeksai
	for (int i = 0; i < L; i++) {
		s[i] = 0;
		for (int j = i; j > 0; j--) {
			s[i] += l[j - 1];
		}
	}

	//Bias neuronai
	for (int i = 0; i < L - 1; i++) {
		a_arr[s[i + 1] - 1] = 1;
	}


	//Svoriu kiekiai l-ame sluoksnyje
	for (int i = 0; i < L - 1; i++) {
		W[i] = l[i] * (l[i + 1] - 1);
		sw[i] = 0;
		if (i != 0) {
			for (int j = 0; j < i; j++) {
				sw[i] += W[j];
			}
		}
  }


  if (pFile==NULL) {
    for (int i = 0; i < L - 1; i++)
      for (int j = 0; j < W[i]; j++) {
        w_arr[sw[i] + j] =(rnd->next()*2-1); // (double)rand() / double(RAND_MAX);
        dw_arr[sw[i] + j] = 0.0;
    }
  }
  else {
  //  readf_Network(pFile);
  }

}


void AnnSerialDBL::feedForward(double *h_input,double *a, double *b){
	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_arr[i] = a[i];
	}

  for(int i=0; i<M;i++){
    ah_arr[i] = h_input[i];
  }

	for (int j = 0; j < cTopology->obtainNeuronCount(); j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	for (int i = 0; i<cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++)
		b[i] = a_arr[s[L - 1] + i];
}

void AnnSerialDBL::calc_feedForward(){
  for(int i = 0; i < l[1] - 1; i++){
    for(int j = 0; j < M; j++){
      z_arr[s[1] + i] += ah_arr[j] * wh_arr[j*M + i];
    }
  }
	for (int i = 0; i < L - 1; i++) {//per sluoksnius einu+
		for (int j = 0; j < l[i]; j++) { //kiek neuronu sluoksnyje+
			for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z+
				z_arr[s[i + 1] + k] += w_arr[sw[i] + k + j*(l[i + 1] - 1)] * a_arr[s[i] + j];
			}
		}
		for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z
			a_arr[s[i + 1] + k] = f(z_arr[s[i + 1] + k]);
		}
	}
}

double AnnSerialDBL::f(double x) {
	//return atan(x)/M_PI + 0.5;
	double y = 1 + exp(-x);
	return 1 / y;
}



double* AnnSerialDBL::getWeights(){
	return w_arr;
}

double* AnnSerialDBL::getDWeights(){
	return dw_arr;
}

double* AnnSerialDBL::getHWeights(){
	return wh_arr;
}

double* AnnSerialDBL::getDHWeights(){
	return dwh_arr;
}

double* AnnSerialDBL::getA(){
	return a_arr;
}

Topology* AnnSerialDBL::getTopology(){
  return cTopology;
}
