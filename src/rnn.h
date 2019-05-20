#ifndef RNN_HEADER
#define RNN_HEADER

//#include <helper_cuda.h>
#include "ann.h"

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
  Derivatives **cderiv;
};

struct ErrorDerivatives {
  double *v;
  double *vh;
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
    int V;

    AnnSerial** anns;
    AnnSerial* ann_forget; // 0
    AnnSerial* ann_input; // 1
    AnnSerial* ann_gate; // 2
    AnnSerial* ann_output; // 3

    double* c_current;
    double* c_new;
    double* h_current;
    double* h_new;
    double* b;
    double** a_outputs;

    Derivatives ***aderiv;




  public:

    RnnCell(int M, Topology **top) {
      prepare(M, top);
      init(NULL);
    };

    RnnCell(string filename) {
        // FILE * p1File;
        // p1File = fopen(filename.c_str(), "rb");
        // Topology *top=new Topology();
        // top->readTopology(p1File);
        // prepare(top);
        // init(p1File);
        // fclose (p1File);
    };

     void train(double *a, double *b, double alpha, double eta){};
  	 void feedForward(double *h_in, double *c_in,double *a, double *c_out, double *h_out);
     void backPropagation( RnnDerivatives *deriv_in, RnnDerivatives *deriv_out);



    //
    AnnSerial* getANN(int v);




    void destroy();
  private:
    void prepare(int M, Topology **top);
    void init(FILE *pFile);

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
    double *c_in;
    double *c_out;

    ErrorDerivatives** errDeriv;
    RnnDerivatives **rnnDeriv;



  public:
    Rnn(int I, int M, RnnCell *rnnCell);

    DataNode* feedForward(DataNode* input, OutputLimit *outputLimit);
    bool backPropagation(DataNode* input, DataNode* output, OutputLimit *outputLimit, double &error);

    void updateWeights(double alpha, double eta);
    void resetErrorDerivatives();



  private:

    RnnDerivatives* allocateRnnDerivatives(RnnDerivatives* deriv);
    //RnnDerivatives* deallocateRnnDerivatives(RnnDerivatives* deriv);
    void initRnnDerivatives(RnnDerivatives* deriv);
    void copyRnnDerivatives(RnnDerivatives* deri_b, RnnDerivatives* deriv_a);
    void copyVector(double* vec_b, double *vec_a, int n);

    void sumErrorDerivatives(double *h, Derivatives **hderiv, double *y);
    double calcError(double *h, double *y);


};

double f(double x);
double f_deriv(double x);

double f_tanh(double x);
double f_tanh_deriv(double x);



#endif /* RNN_HEADER */
