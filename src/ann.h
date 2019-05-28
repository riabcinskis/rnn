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

#include "rnn.h"

struct ErrorDerivatives;

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
    //-ml : vector<int>
	public:
		Topology();
    //+Topology()
		~Topology();
    //+~Topology()

		void addLayer(int size);
    //+void addLayer(int size) : void

		int getLayerCount();
    //+getLayerCount() : int
		int getLayerSize(int index);
    //+getLayerSize(index : int) : int

		int obtainNeuronCount();
    //+obtainNeuronCount() : int
		int obtainWeightCount();
    //+obtainWeightCount() : int

		int getInputNeuronCount();
    //+getInputNeuronCount() : int
		int getOutputNeuronCount();
    //+getOutputNeuronCount() : int

    void printTopology(FILE *file);
    //+printTopology(file : FILE) : void
    void readTopology(FILE *file);
    //+readTopology(file : FILE) : void
};

struct Derivatives{
  double *v;
  //+v : double[]
  double *vh;
  //+vh : double[]
};

// Derivatives *a = new Derivatives;

// Derivatives *a = new Derivatives[4];
// for(int v = 0; v < 4; v++) a[v] = new Derivatives;


template <typename T>
class AnnBase {
  public:
  	//virtual void train(T *a, T *b, T alpha, T eta) = 0;

    virtual void feedForward(T *h_input, T *a, T *b) = 0; // MB: h, a->x, b->a
    virtual void backPropagation(Derivatives **deriv_in, Derivatives **deriv_out);
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


  int * nG;
  int * sG;//pazieti kurie int kurie double
  double * G;

  double (*f)(double);
  double (*f_deriv)(double);


public:

  AnnSerial(int V, int u, int M, Topology **top,  double (*f)(double), double (*f_deriv)(double));
  //+AnnSerial(V : int, u : int, M : int, top : Topology[],  (*f)(double) : double, (*f_deriv)(double) : double)

  AnnSerial(int V, int u, int M, double (*f)(double), double (*f_deriv)(double), string filename) {
      // FILE * p1File;
      // p1File = fopen(filename.c_str(), "rb");
      // Topology *top=new Topology();
      // top->readTopology(p1File);
      // prepare(top);
      // init(p1File);
      // fclose (p1File);
  };

  void destroy();
  //+destroy() : void

  void feedForward(double *h_input, double *a, double *b); // ...
  //+feedForward(h_input : double[], a : double[], b : double[]) : void
  void backPropagation(Derivatives **deriv_in, Derivatives **deriv_out);
  //+backPropagation(deriv_in : Derivatives[], deriv_out : Derivatives[]) : void

  void updateWeights(ErrorDerivatives *errDeriv, double alpha, double eta);
  //+updateWeights(errDeriv : ErrorDerivatives, alpha : double, eta : double) : void

private:

  void prepare(Topology **top);
  //-prepare(top : Topology[]) : void
  void init(Topology **top,FILE *pFile);
  //-init(top : Topology[],pFile : FILE) : void

  void calc_feedForward();
  //-calc_feedForward() : void
  void copyOutput(double *a);
  //-copyOutput(a : double[]) : void

  void calcG();
  //-calcG() : void
  void calcDerivatives(int v, Derivatives *deriv_h, Derivatives *deriv_a);
  //-calcDerivatives(v : int, deriv_h : Derivatives, deriv_a : Derivatives) : void

  int obtainGCount(int L);
  //-obtainGCount(L : int) : int
  int layerToGIndex(int L, int l);
  //-layerToGIndex(L : int, l : int) : int

  double d(int i, int j);
  //-d(i : int, j : int) : double




public:
  int vi(int v, int s, int i, int j, int k);
  //+vi(v : int, s : int, i : int, j : int, k : int) : int
  int vhi(int v, int i, int j, int k);
  //+vhi(v : int, i : int, j : int, k : int) : int

  int vi(int v, int s, int i, int j);
  //+vi(v : int, s : int, i : int, j : int) : int
  int vhi(int v, int i, int j);
  //+vhi(v : int, i : int, j : int) : int


  void setWeights(double *W, double *Wh);
  //+setWeights(W : double[], Wh : double[]) : void


  double* getWeights();
  //+getWeights() : double[]
  double* getHWeights();
  //+getHWeights() : double[]
  double* getDWeights();
  //+getDWeights() : double[]
  double* getDHWeights();
  //+getDHWeights() : double[]
  double* getA();
  //+getA() : double[]
  double getOutput(int k);
  //+getOutput(k : int) : double

  Topology* getTopology();
  //+getTopology() : Topology
};

//
// Global functions
//

void obtainSW(Topology *top, int *sW);


#endif /* */
