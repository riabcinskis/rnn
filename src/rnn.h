#ifndef RNN_HEADER
#define RNN_HEADER

#define RNN_FULL_BACKPROPAGATION 0
#define RNN_APPROX_BACKPROPAGATION 1


//#include <helper_cuda.h>
#include "ann.h"
#include "rnn.h"



#include <cmath>

class Topology;
class AnnSerial;
struct Derivatives;

/* Class definitions here. */
void run_cuda_sample();

class RnnConfig{
private:
  Topology** cTopology;
  int mM;


public:
  void setTopologies(Topology **top);
  void setM(int M);


  Topology** getTopologies();
  int getM();

};



struct RnnDerivatives {
  Derivatives **hderiv;
  //+hderiv : Derivatives[]
  Derivatives **cderiv;
  //+cderiv : Derivatives[]
};

struct ErrorDerivatives {
  double *v;
  //+v : double[]
  double *vh;
  //+vh : double[]
};

template <typename T>
class RnnBase {
  public:
  	 virtual void train(T *a, T *b, T alpha, T eta) = 0;
  	 virtual void feedForward(T *h_in, T *c_in,T *a, T *c_out, T *h_out) = 0;


  private:
     virtual	void init(FILE *pFile)=0;
};


class RnnCell {
  private:

    int M;
    //-M : int
    int V;
    //-V : int

    AnnSerial** anns;
    //-anns : AnnSerial[]
    AnnSerial* ann_forget; // 0

    AnnSerial* ann_input; // 1
    AnnSerial* ann_gate; // 2
    AnnSerial* ann_output; // 3

    double* c_current;
    //-c_current : double[]
    double* c_new;
    //-c_new : double[]
    double* h_current;
    //-h_current : double[]
    double* h_new;
    //-h_new : double[]
    double* b;
    //-b : double[]
    double** a_outputs;
    //-a_outputs : double[][]

    Derivatives ***aderiv;
    //-aderiv : Derivatives[][]




  public:

    RnnCell(int M, Topology **top) {
      prepare(M, top);
      init(NULL);
    };
    //+RnnCell(M : int, top : Topology[])


    RnnCell(int M, string filename);



  	 void feedForward(double *h_in, double *c_in,double *a, double *c_out, double *h_out);
     //+feedForward(h_in : double[], c_in : double[],a : double[], c_out : double[], h_out : double[]) : void
     void backPropagation( RnnDerivatives *deriv_in, RnnDerivatives *deriv_out);
     //+backPropagation( deriv_in : RnnDerivatives, deriv_out : RnnDerivatives) : void

    //
    AnnSerial* getANN(int v);
    //+getANN(v : int) : AnnSerial

    void destroy();
    //+destroy() : void

    void printf_Network(string output_filename);

    //+printf_Network(output_filename : string) : void
  private:
    void prepare(int M, Topology **top);
    //-prepare(M : int, top : Topology[]) : void
    void init(FILE *pFile);
    //-init(pFile : FILE) : void

};


class OutputLimit {
  public:
    virtual void reset() = 0;
    virtual bool check(double *vec) = 0;
};

class SecondMarkLimit : public OutputLimit {
  private:
    int M;
    int count;
    int markIndex;

  public:
    SecondMarkLimit(int markIndex, int M);

    void reset();
    bool check(double *vec);
};

struct DataNode{
  DataNode(int M);
  double* vec;
  DataNode *next;
};


class Rnn {
  private:
    int I;
    int M;
    int V;

    RnnCell* cRnnCell;


    double *h_in;
    double *h_out;
    double* c_in;
    double *c_out;

    ErrorDerivatives** errDeriv;
    RnnDerivatives **rnnDeriv;

    int impl;

  public:
    Rnn(int I, int M, RnnCell *rnnCell, int impl);
    Rnn(int I, int M, RnnCell *rnnCell);

    DataNode* feedForward(DataNode* input, OutputLimit *outputLimit);
    bool backPropagation(DataNode* input, DataNode* output, OutputLimit *outputLimit, double &error);

    void updateWeights(double alpha, double eta);
    //+updateWeights(alpha : double, eta : double) : void
    void resetErrorDerivatives();
    //+resetErrorDerivatives() : void


    RnnCell* getRnnCell(){
      return cRnnCell;
    }

    //+getRnnCell() : RnnCell

  private:

    RnnDerivatives* allocateRnnDerivatives(RnnDerivatives* deriv);
    //-allocateRnnDerivatives(deriv : RnnDerivatives) : RnnDerivatives

    //RnnDerivatives* deallocateRnnDerivatives(RnnDerivatives* deriv);
    void resetHDerivatives(Derivatives** hderiv);
    //-resetHDerivatives(hderiv : Derivatives[]) : void
    void resetCDerivatives(Derivatives** cderiv);


    void copyRnnDerivatives(RnnDerivatives* deri_b, RnnDerivatives* deriv_a);
    //-copyRnnDerivatives(deri_b : RnnDerivatives, deriv_a : RnnDerivatives) : void
    void copyVector(double* vec_b, double *vec_a, int n);
    //-copyVector(vec_b : double[], vec_a : double[], n : int) : void

    void sumErrorDerivatives(double *h, Derivatives **hderiv, double *y);
    //-sumErrorDerivatives(h : double, hderiv : Derivatives[], y : double[]) : void
    double calcError(double *h, double *y);
    //-calcError(h : double[], y : double[]) : double


};

double f(double x);
double f_deriv(double x);

double f_tanh(double x);
double f_tanh_deriv(double x);



#endif /* RNN_HEADER */
