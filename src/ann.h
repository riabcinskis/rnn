#ifndef ANN_HEADER
#define ANN_HEADER

#include <cmath>
#include <cstdlib>

#include <cmath>
#include <math.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <array>
#include <assert.h>

#include <random>

using namespace std;

//
// Random
//
class Random {
  private:
    std::mt19937 *mGen;
    std::uniform_real_distribution<double> *mDist;

  public:
    Random();
    double next();
    int nextInt(int min, int max);
    bool nextBool();
};


class Topology {
	private:
		std::vector<int> *ml;
	public:
		Topology();
		~Topology();
		void addLayer(int size);

		int getLayerCount();
		int getLayerSize(int index);

		int obtainNeuronCount();
		int obtainWeightCount();

		int getInputNeuronCount();
		int getOutputNeuronCount();

    // void printTopology(FILE *file);
    // void readTopology(FILE *file);
};

struct Derivatives{
  double *v;
  double *vh;
};

// Derivatives *a = new Derivatives;

// Derivatives *a = new Derivatives[4];
// for(int v = 0; v < 4; v++) a[v] = new Derivatives;


template <typename T>
class AnnBase {
  public:
  	//virtual void train(T *a, T *b, T alpha, T eta) = 0;

    virtual void feedForward(T *h_input, T *a, T *b) = 0; // MB: h, a->x, b->a
    virtual void backPropagation(Derivatives **deriv_in, Derivatives **deriv_out, double *a);
    virtual void destroy() = 0;
  	//virtual T obtainError(T *b) = 0;

  //	virtual void print_out() = 0;

  private:
    virtual void prepare(Topology **top, int M)=0; // MB: M
    virtual	void init(FILE *pFile)=0; // MB: pFile neraikia kol kas
    virtual void reset()=0;
    virtual void calc_feedForward()=0;
    virtual void copyOutput(double *a)=0;
};



class AnnSerial{ // -> AnnSerial
private:
  Topology* cTopology;
  int V; // total number of ANNs
  int u; // index of ANN
  int L;
  int M;
  int * l; // neuronu skaičius sluoksnyje
  int **vl;
  int *vL;
  int * s; // sluoksnio pradzios indeksai
  double * a_arr;
  double * ah_arr;
  double * z;
  int * sW;
  int ** vsW;

  double * W;
  double * dW;
  double * Wh;
  double * dWh;


  double * nG;
  int * sG;//pazieti kurie int kurie double
  double * G;

  double (*f)(double);
  double (*f_deriv)(double);


public:

  AnnSerial(int V, int u, int M, Topology **top,  double (*f)(double), double (*f_deriv)(double));

  void destroy();

  void feedForward(double *h_input, double *a, double *b); // ...
  void backPropagation(Derivatives **deriv_in, Derivatives **deriv_out, double *a);


private:

  void prepare(Topology **top);
  void init(Topology **top,FILE *pFile);

  void reset();
  void calc_feedForward();
  void copyOutput(double *a);

  void calcG();
  void calcDerivatives(int v, Derivatives *deriv_h, Derivatives *deriv_a);

  int obtainGCount(int L);
  int layerToGIndex(int L, int l);


  int vi(int v, int s, int i, int j, int k);
  int vhi(int v, int i, int j, int k);

  double d(int i, int j);


public:
  void setWeights(double *W, double *Wh);


  double* getWeights();
  double* getHWeights();
  double* getDWeights();
  double* getDHWeights();
  double* getA();
  Topology* getTopology();
};

//
// Global functions
//

void obtainSW(Topology *top, int *sW);


#endif /* */
