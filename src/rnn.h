#ifndef RNN_HEADER
#define RNN_HEADER

//#include <helper_cuda.h>
#include "ann.h"

#include <cmath>


/* Class definitions here. */
void run_cuda_sample();

class rnnConfig{
private:
  Topology** cTopology;
  int mM;
  int mV;

public:
  void setTopology(Topology **top);
  void setM(int M);
  void setV(int V);

  Topology** getTopology();
  int getM();
  int getV();
};

template <typename T>
class RnnBase {
  public:
  	 virtual void train(T *a, T *b, T alpha, T eta) = 0;
  	 virtual void feedForward(T *h_in, T *c_in,T *a, T *c_out, T *h_out) = 0;


  private:
     virtual	void init(FILE *pFile)=0;
};


class RnnSerial : public RnnBase<double> {
  private:

    int M;
    int V;

    AnnSerial ** anns;

    double * c_current;
    double * c_new;
    double * h_current;
    double * h_new;
    double * b;
    double ** a_outputs;

  public:
  	 void train(double *a, double *b, double alpha, double eta){};
  	 void feedForward(double *h_in, double *c_in,double *a, double *c_out, double *h_out);

    RnnSerial(rnnConfig *mRnnConf) {
      prepare(mRnnConf);
      init(NULL);
    };

  	RnnSerial(string filename) {
        // FILE * p1File;
        // p1File = fopen(filename.c_str(), "rb");
        // Topology *top=new Topology();
        // top->readTopology(p1File);
        // prepare(top);
        // init(p1File);
        // fclose (p1File);
    };

    //
    AnnSerial* getANN(int v);


    void backPropagation(double *h_in, double *a, Derivatives **deriv_in, Derivatives **deriv_out);


    void destroy();
  private:
    void prepare(rnnConfig *mRnnConf);
    void init(FILE *pFile);

};

double f(double x);
double f_deriv(double x);

double f_tanh(double x);
double f_tanh_deriv(double x);



#endif /* RNN_HEADER */
