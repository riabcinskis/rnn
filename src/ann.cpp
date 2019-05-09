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
// AnnSerialDBL
//

AnnSerialDBL(int u, int M, Topology **top, double (*f)(double), double (*f_deriv)(double)){
  this->u = u;
  this->M = M;
  cTopology = top[u];
  L = cTopology->getLayerCount();

  assert(M == top->getOutputNeuronCount());

  prepare(top);
  init(NULL);
}

int AnnSerialDBL::obtainGCount(int L){
  int G_count = 0;
  int remaining_L = L;
  while(remaining_L >= 3){
    G_count++;
    remaining_L -= 2;
  }
  return G_count;
}

void AnnSerialDBL::obtainSW(Topology *top, int *sW){
  int nW;
  sW[i] = 0;
  for (int i = 1; i < L-1; i++) {
		nW = (top->getLayerSize(i-1)+1)*top->getLayerSize(i); //l[i] * (l[i + 1] - 1);
    sW[i] = sW[i-1] + nW;
  }
}

void AnnSerialDBL::prepare(Topology **top){

	l = new int[cTopology->getLayerCount()];
	s = new int[cTopology->getLayerCount()];

	int neuronCount = cTopology->obtainNeuronCount();
	int weightCount = cTopology->obtainWeightCount();

	a_arr = new double[neuronCount];
	z_arr = new double[neuronCount];

  ah_arr = new double[M];

  vsW = new int*[4];
  for(int v = 0; v < 4; v++)
    vsW[v] = new int[top[v]->getLayerCount()-1];

	sW = new int[cTopology->getLayerCount()-1];


	w_arr = new double[weightCount];
	dw_arr = new double[weightCount];

  int second_layer_size = cTopology->getLayerSize(2);
  int h_weightCount = second_layer_size * M;

  wh_arr = new double[h_weightCount];
  dwh_arr = new double[h_weightCount];

  int G_count = obtainGCount(L);


  nG = new double[G_count];
  sG = new double[G_count];


  int count = 0;
  int l2;
  for(int gl = 0; gl > G_count; gl++){
    l2 = L - 2*gl - 1;
    nG[gl] = cTopology->getLayerSize(l2)*(cTopology->getLayerSize(l2 - 2)+1);
    count += nG[gl];
  }

  sG[0] = 0;
  for(int gl = 1; gl > G_count; gl++)
    sG[gl] = sG[gl-1] + nG[gl-1];

  G = new double[count];



	//gjl = new double[neuronCount];
}

void AnnSerialDBL::init(FILE * pFile=NULL){

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


  obtainSW(top[u], sW);
  for(int v = 0; v < 4; v++){
    obtainSW(top[v], vsW[v]);
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

  delete rnd;

}

void AnnSerialDBL::reset(){



}

void AnnSerialDBL::copyOutput(double *a){
  for (int i = 0; i<cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++)
		a[i] = a_arr[s[L - 1] + i];
}

void AnnSerialDBL::feedForward(double *h_input, double *a, double *b){

  for (int i = 0; i < cTopology->getLayerSize(0); i++) {
    a_arr[i] = a[i];
  }

  for(int i=0; i<M;i++){
    ah_arr[i] = h_input[i];
  }

	calc_feedForward();

	copyOutput(b);
}

void AnnSerialDBL::calc_feedForward(){

  for (int j = 0; j < cTopology->obtainNeuronCount(); j++) {
		z_arr[j] = 0;
	}

  for(int k = 0; k < l[1] - 1; k++){
    for(int j = 0; j < M; j++){
      z_arr[s[1] + k] += ah_arr[j] * wh_arr[j*(l[1]-1) + k];
    }
  }
	for (int i = 0; i < L - 1; i++) {//per sluoksnius einu+
    for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z+
		  for (int j = 0; j < l[i]; j++) { //kiek neuronu sluoksnyje+
				z_arr[s[i + 1] + k] += w_arr[sw[i] + j*(l[i + 1] - 1) + k] * a_arr[s[i] + j];
			}
		}
		for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z
			a_arr[s[i + 1] + k] = f(z_arr[s[i + 1] + k]);
		}
	}
}

void AnnSerialDBL::backPropagation(Derivatives *deriv_in, Derivatives *deriv_out, double *a){

  calc_feedForward();
  calcG();

  for(int v = 0; v < 4; v++)
  calcDerivatives(1, deriv_in[v],  deriv_out[v]);

  copyOutput(a);

}

void AnnSerialDBL::calcG(){
  int G_count = obtainGCount(L);

  for(int gl = 0; gl > G_count; gl++){
    l2 = L - 2*gl - 1;

    double sum = 0;

    if(l2==2){

      for(int p = 0; p < M; p++){
        for(int k = 0; k < l[l2]-1; k++){

          for(int n = 0; n < l[l2-1]-1; n++){
            sum += Wh[(l[l2-1]-1)*p + n]*W[sW[l2-1] + (l[l2]-1)*n + k]*f_deriv(z[s[l2-1]+n]);
          }
        }

        G[sG[gl] + (l[l2]-1)*p + k] = f_deriv(z[s[l2] + k])*sum;
      }

    }else{
      for(int p = 0; p < l[l2-2]; p++){
        for(int k = 0; k < l[l2]-1; k++){

          for(int n = 0; n < l[l2-1]-1; n++){
            sum += W[sW[l2-2] + (l[l2-1]-1)*p + n]*W[sW[l2-1] + (l[l2]-1)*n + k]*f_deriv(z[s[l2-1]+n]);
          }

          G[sG[gl] + (l[l2]-1)*p + k] = f_deriv(z[s[l2] + k])*sum;
        }
      }
    }


  }
}

double AnnSerialDBL::d(int i, int j){
  if(i == j) return 1.0;
  return 0.0;
}


void AnnSerialDBL::calcDerivatives(int v, Derivatives *deriv_in, Derivatives *deriv_out){
  // deriv_in->v[vi(...)];
  // deriv_in->vh[vhi(...)];

}

int AnnSerialDBL::vi(int v, int s, int i, int j, int k){
  return (vsW[v][s] + i*(vl[v][s+1]-1) + j)*M + k;
}

int AnnSerialDBL::vhi(int v, int i, int j, int k){
  return  (i*(vl[v][1]-1) + j)*M + k;
}



// double AnnSerialDBL::f(double x) {
// 	//return atan(x)/M_PI + 0.5;
// 	double y = 1 + exp(-x);
// 	return 1 / y;
// }

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
