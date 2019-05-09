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

    virtual void feedForward(T *h_input,T *a, T *b) = 0; // MB: h, a->x, b->a
    virtual void backPropagation(Derivatives *deriv_in, Derivatives *deriv_out);
    virtual void destroy() = 0;
  	//virtual T obtainError(T *b) = 0;

  //	virtual void print_out() = 0;

  private:
    virtual void prepare(Topology **top, int mM)=0; // MB: M
    virtual	void init(FILE *pFile)=0; // MB: pFile neraikia kol kas
    virtual void reset()=0;
    virtual void calc_feedForward()=0;
    virtual void copyOutput(double *a)=0;
};


class AnnSerialDBL{ // -> AnnSerial
private:
  Topology* cTopology;

  int u; // index of ANN
  int L;
  int M;
  int * l; // neuronu skaičius sluoksnyje
  int **vl;
  int * s; // sluoksnio pradzios indeksai
  double * a_arr;
  double * ah_arr;
  double * z_arr;
  int * sW;
  int ** vsW;

  double * w_arr;
  double * dw_arr;
  double * wh_arr;
  double * dwh_arr;


  int * nG;
  int * sG;
  double * G;

  double (*f)(double);
  double (*f_deriv)(double);


public:

  AnnSerialDBL(int u, int mM, Topology **top,  double (*f)(double), double (*f_deriv)(double));

  void setWeights(double *t_w_arr, double *t_wh_arr){ // MB: naudojam this->w_arr = w_arr, iskelti i ann.cpp
    w_arr = t_w_arr;
    wh_arr = t_wh_arr;
  };

  void feedForward(double *h_input, double *a, double *b); // ...
  void backPropagation(Derivatives *deriv_in, Derivatives *deriv_out);


  double* getWeights();
  double* getHWeights();
  double* getDWeights();
  double* getDHWeights();
  double* getA();
  Topology* getTopology();


private:

  void prepare(Topology **top);
  void init(FILE *pFile);
  void reset();
  void calc_feedForward();
  void copyOutput(double *a);

  void calcG();
  void calcDerivatives(int v, double *deriv_h, double *deriv_a);

  int obtainGCount(int L);
  void obtainSW(Topology *top, int *sW);
  int vi(int v, int s, int i, int j, int k);
  int vhi(int i, int j, int k);

  double d(int i, int j);





};


#endif /* */
