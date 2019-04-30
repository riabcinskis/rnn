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

template <typename T>
class AnnBase {
  public:
  	virtual void train(T *a, T *b, T alpha, T eta) = 0;
  	virtual void feedForward(T *h_input,T *a, T *b) = 0;
  	virtual void destroy() = 0;
  	virtual T obtainError(T *b) = 0;

  //	virtual void print_out() = 0;

  private:
    virtual void prepare(Topology *top, int mM) = 0;
    virtual	void init(FILE *pFile)=0;
  	virtual void calc_feedForward() = 0;
};


class AnnSerialDBL{
private:
  Topology* cTopology;

  int L;
  int M;
  int * l;
  int * s;
  double * a_arr;
  double * ah_arr;
  double * z_arr;
  int * W;
  int * sw;
  double * w_arr;
  double * dw_arr;
  double * wh_arr;
  double * dwh_arr;


public:
  void feedForward(double *h_input,double *a, double *b);

  void setWeights(double *t_w_arr, double *t_wh_arr){
    w_arr = t_w_arr;
    wh_arr = t_wh_arr;
  };


  AnnSerialDBL(Topology *top, int mM) {
    prepare(top,mM);
    init(NULL);
  };


  double* getWeights();
  double* getHWeights();
  double* getDWeights();
  double* getDHWeights();
  double* getA();
  Topology* getTopology();


private:
  void prepare(Topology *top, int M);
  void init(FILE *pFile);

    	void calc_feedForward();

  double f(double x);
  double f_deriv(double x);

};

#endif /* */
