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
  count--;
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
// AnnSerial
//

AnnSerial::AnnSerial(int V, int u, int M, Topology **top, double (*f)(double), double (*f_deriv)(double)){
  this->V = V;
  this->u = u;
  this->M = M;
  cTopology = new Topology();
    cTopology = top[u];

  L = cTopology->getLayerCount();

  this->f=(*f);
  this->f_deriv=(*f_deriv);

  assert(M == top[u]->getOutputNeuronCount());
  prepare(top);

  init(top,NULL);
}

void AnnSerial::destroy(){
	delete[] l;
	l = NULL;

  delete [] vl;
  vl = NULL;

  delete[] vL;
  vL= NULL;

	delete[] s;
	s = NULL;

	delete[] a_arr;
	a_arr = NULL;

  delete[] ah_arr;
  ah_arr = NULL;

	delete[] z;
	z = NULL;

	delete[] sW;
	sW = NULL;

  delete [] vsW;
  vsW = NULL;

	delete[] W;
	W = NULL;
	delete[] dW;
	dW = NULL;

  delete[] Wh;
  Wh = NULL;
  delete[] dWh;
  dWh = NULL;

  delete[] nG;
  nG = NULL;
  delete[] sG;
  sG = NULL;


	// delete[] t_arr;
	// t_arr = NULL;

	// delete[] gjl;
	// gjl = NULL;
}

void AnnSerial::prepare(Topology **top){

  vL = new int[V];
  vl = new int*[V];
  for(int i = 0; i < V; i++){
    vl[i] = new int[top[u]->getLayerCount()];
  }
	l = new int[cTopology->getLayerCount()];
	s = new int[cTopology->getLayerCount()];

	int neuronCount = cTopology->obtainNeuronCount();
	int weightCount = cTopology->obtainWeightCount();

	a_arr = new double[neuronCount];
	z = new double[neuronCount];

  ah_arr = new double[M];

  vsW = new int*[V];
  for(int v = 0; v < V; v++)
    vsW[v] = new int[top[v]->getLayerCount()-1];

	sW = new int[cTopology->getLayerCount()-1];

	W = new double[weightCount];
	dW = new double[weightCount];

  int second_layer_size = cTopology->getLayerSize(2);
  int h_weightCount = second_layer_size * M;

  Wh = new double[h_weightCount];
  dWh = new double[h_weightCount];

  int G_count = obtainGCount(L);


  nG = new int[G_count];
  sG = new int[G_count];

  int count = 0;
  int l2;
  for(int gl = 0; gl < G_count; gl++){
    l2 = L - 2*gl - 1;
    nG[gl] = cTopology->getLayerSize(l2)*(cTopology->getLayerSize(l2 - 2));
    count += nG[gl];
  }


  for(int gl = 0; gl < G_count; gl++){
    if(gl == 0)
      sG[0] = 0;
    else
      sG[gl] = sG[gl-1] + nG[gl-1];
  }

  G = new double[count];


	//gjl = new double[neuronCount];
}

void AnnSerial::init(Topology **top,FILE * pFile=NULL){

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

  for(int i = 0; i < V; i++){
    vL[i] = top[i]->getLayerCount()-1;
  }

  for(int i = 0; i < V; i++){
    for(int j = 0; j < vL[i]; i++)
      vl[i][j] = top[i]->getLayerSize(j)+1;
  }



	//Svoriu kiekiai l-ame sluoksnyje


  obtainSW(top[u], sW);
  for(int v = 0; v < V; v++){
    obtainSW(top[v], vsW[v]);
  }

  if (pFile==NULL) {
    for (int i = 0; i < L - 1; i++)
      for (int j = 0; j < W[i]; j++) {
        W[sW[i] + j] =(rnd->next()*2-1); // (double)rand() / double(RAND_MAX);
        dW[sW[i] + j] = 0.0;
    }
  }
  else {
  //  readf_Network(pFile);
  }

  delete rnd;

}

void AnnSerial::reset(){



}

void AnnSerial::feedForward(double *h_input, double *a, double *b){

  for (int i = 0; i < cTopology->getLayerSize(0); i++) {
    a_arr[i] = a[i];
  }
  a_arr[cTopology->getLayerSize(0)] = 1.0;

  for(int i=0; i<M;i++){
    ah_arr[i] = h_input[i];
  }

	calc_feedForward();

	copyOutput(b);
}

void AnnSerial::calc_feedForward(){

  for (int j = 0; j < cTopology->obtainNeuronCount(); j++) {
		z[j] = 0;
	}
  for(int k = 0; k < l[1] - 1; k++){
    for(int j = 0; j < M; j++){
      z[s[1] + k] += ah_arr[j] * Wh[j*(l[1]-1) + k];
    }
  }
	for (int i = 0; i < L - 1; i++) {//per sluoksnius einu+
    for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z+
		  for (int j = 0; j < l[i]; j++) { //kiek neuronu sluoksnyje+
				z[s[i + 1] + k] += W[sW[i] + j*(l[i + 1] - 1) + k] * a_arr[s[i] + j];
			}
		}
		for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z
			a_arr[s[i + 1] + k] = f(z[s[i + 1] + k]);
		}
	}
}

void AnnSerial::copyOutput(double *a){
  for (int i = 0; i<cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++)
		a[i] = a_arr[s[L - 1] + i];
}

void AnnSerial::backPropagation(Derivatives **deriv_in, Derivatives **deriv_out){

  calcG();

  for(int v = 0; v < V; v++)
    calcDerivatives(v, deriv_in[v],  deriv_out[v]);
}

void AnnSerial::calcG(){
  int G_count = obtainGCount(L);
  int l2;
  for(int gl = 0; gl < G_count; gl++){
    l2 = L - 2*gl - 1;

    double sum = 0;

    if(l2==2){

      for(int p = 0; p < M; p++){
        for(int k = 0; k < l[l2]-1; k++){

          for(int n = 0; n < l[l2-1]-1; n++){
            sum += Wh[(l[l2-1]-1)*p + n]*W[sW[l2-1] + (l[l2]-1)*n + k]*f_deriv(z[s[l2-1]+n]);
            //pakeiciau W i Wh
            //sum += Wh[M*p + n]*W[sW[l2-1] + (l[l2]-1)*n + k]*f_deriv(z[s[l2-1]+n]);
          }

          G[sG[gl] + (l[l2]-1)*p + k] = f_deriv(z[s[l2] + k])*sum;
        }

      //  G[sG[gl] + (l[l2]-1)*p + k] = f_deriv(z[s[l2] + k])*sum;
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

void AnnSerial::calcDerivatives(int v, Derivatives *deriv_in, Derivatives *deriv_out){
  // deriv_in->v[vi(...)];
  // deriv_in->vh[vhi(...)];

  int max_layer_size = 0;
  for(int i = 0; i < L; i++)
    max_layer_size = l[i] > max_layer_size? l[i]: max_layer_size;

  // vec0 -> vec1
  double *vec0 = new double[max_layer_size];
  double *vec1 = new double[max_layer_size];


  int start_l = L%2 == 1 ? 0 : 1;


    for(int ss = 0; ss < vL[v]-1; ss++){//kiek sluoksniu tinkle
      for(int wi = 0; wi < vl[v][ss]; wi++){//kiek neuronu
        for(int wj = 0; wj < vl[v][ss+1]-1; wj++){//kiek sekanciame neuronu
          for(int ll = start_l; ll <= L; ll=ll+2){//per G reiksmes  cia gal maziau tik

            if(ll == 0){

              for(int k = 0; k < M; k++)
                vec1[k] = deriv_in->v[vi(v, ss, wi, wj, k)];

            }else if(ll == 1){

              for(int k = 0; k < l[ll]-1;k++){
                double sum = 0;
                for(int n = 0; n < M; n++)
                  //sum += W[sW[ll] + l[ll]*n + k]*deriv_in->v[vi(v, ss, wi, wj, n)];
                  sum += Wh[l[ll]*n + k]*deriv_in->v[vi(v, ss, wi, wj, n)];
                if(u == v /*gal sita&& ss == ll*/) sum += a_arr[wi];
                vec1[k] = f_deriv(z[s[ll] + k])*sum;

              }


            }else {


              int gl = layerToGIndex(L, ll);
              double sum = 0;
              int to_p = ll == 2 ? M : l[ll-2];//-1 gal


              for(int k = 0; k < l[ll]-1;k++){
                for(int p = 0; p < to_p; p++){
                  sum += vec0[p]*G[sG[gl] + (l[ll]-1)*p + k];
                }


                if(u == v){
                  if(ll == ss+1)
                    sum += f_deriv(z[s[ll] + k])*a_arr[s[ll-1] + wi];
                    //int ghh=0;
                  else if(ll == ss+2){
                      double sum2 = 0;
                      for(int n = 0; n < l[ll-1]; n++)
                       sum2 += W[sW[ll-1] + (l[ll]-1)*n + k] * f_deriv(z[s[ll-1] + n]);
                      sum += f_deriv(z[s[ll] + k])*a_arr[s[ll-2] + wi]*sum2;
                  }
                }
              }
            }
            int to_k = ll == 0 ? M : l[ll]-1;
            for(int k = 0; k < to_k; k++) vec0[k] = vec1[k];
          }
          for(int k = 0; k < M /*l[L-1]-1*/; k++)
            deriv_out->v[vi(v, ss, wi, wj, k)] = vec1[k];
        }
      }
    }

      for(int wi = 0; wi < M; wi++){//kiek neuronu
        for(int wj = 0; wj < vl[v][1]-1; wj++){//kiek sekanciame neuronu
          for(int ll = start_l; ll <= L; ll=ll+2){//per G reiksmes  cia gal maziau tik
            if(ll == 0){

              for(int k = 0; k < M; k++)
                vec1[k] = deriv_in->vh[vhi(v, wi, wj, k)];

            }else if(ll == 1){

              for(int k = 0; k < l[ll]-1;k++){
                double sum = 0;
                for(int n = 0; n < M; n++)
                  //sum += W[sW[ll] + l[ll]*n + k]*deriv_in->v[vi(v, ss, wi, wj, n)];
                  sum += Wh[l[ll]*n + k]*deriv_in->vh[vhi(v, wi, wj, n)];
                if(u == v /*gal sita&& ss == ll*/) sum += ah_arr[wi];
                vec1[k] = f_deriv(z[s[ll] + k])*sum;

              }

            }else {


              int gl = layerToGIndex(L, ll);
              double sum = 0;
              int to_p = ll == 2 ? M : l[ll-2];//-1 gal


              for(int k = 0; k < l[ll]-1;k++){
                for(int p = 0; p < to_p; p++){
                  sum += vec0[p]*G[sG[gl] + (l[ll]-1)*p + k];
                }


                if(u == v){
                  // if(ll == 1)
                  //   sum += f_deriv(z[s[ll] + k])*a_arr[s[ll-1] + wi];
                  //   //int ghh=0;
                  // else if(ll == 2){
                  //     double sum2 = 0;
                  //     for(int n = 0; n < l[ll-1]; n++)
                  //      sum2 += W[sW[ll-1] + (l[ll]-1)*n + k] * f_deriv(z[s[ll-1] + n]);
                  //     sum += f_deriv(z[s[ll] + k])*a_arr[s[ll-2] + wi]*sum2
                  // }
                  if(ll == 2){
                    double sum2 = 0;
                    for(int n = 0; n < l[ll-1]; n++)
                      sum2 += W[sW[ll-1] + (l[ll]-1)*n + k] * f_deriv(z[s[ll-1] + n]);
                    sum += f_deriv(z[s[ll] + k])*ah_arr[wi]*sum2;
                  }
                }
              }
            }
            int to_k = ll == 0 ? M : l[ll]-1;
            for(int k = 0; k < to_k; k++) vec0[k] = vec1[k];
          }
          for(int k = 0; k < M /*l[L-1]-1*/; k++)
            deriv_out->vh[vhi(v, wi, wj, k)] = vec1[k];
        }
      }


  delete vec0;
  delete vec1;

}

int AnnSerial::obtainGCount(int L){
  int G_count = 0;
  int remaining_L = L;
  while(remaining_L >= 3){
    G_count++;
    remaining_L -= 2;
  }
  return G_count;
}

int AnnSerial::layerToGIndex(int L, int l){
  int G_count = obtainGCount(L);
  return G_count - (l - l%2) / 2;
}

int AnnSerial::vi(int v, int s, int i, int j, int k){
  return (vsW[v][s] + i*(vl[v][s+1]-1) + j)*M + k;
}

int AnnSerial::vhi(int v, int i, int j, int k){
  return  (i*(vl[v][1]-1) + j)*M + k;
}

double AnnSerial::d(int i, int j){
  if(i == j) return 1.0;
  return 0.0;
}

void AnnSerial::setWeights(double *W, double *Wh){
  this->W = W;
  this->Wh = Wh;
};

double* AnnSerial::getWeights(){
	return W;
}

double* AnnSerial::getDWeights(){
	return dW;
}

double* AnnSerial::getHWeights(){
	return Wh;
}

double* AnnSerial::getDHWeights(){
	return dWh;
}

double* AnnSerial::getA(){
	return a_arr;
}

Topology* AnnSerial::getTopology(){
  return cTopology;
}

//
// Global functions
//


void obtainSW(Topology *top, int *sW){
  int nW;
  sW[0] = 0;
  for (int i = 1; i < top->getLayerCount()-1; i++) {
		nW = (top->getLayerSize(i-1)+1)*top->getLayerSize(i); //l[i] * (l[i + 1] - 1);
    sW[i] = sW[i-1] + nW;
  }
}
