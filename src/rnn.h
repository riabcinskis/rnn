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

public:
  void setTopology(Topology **top);
  void setTopology1(Topology *top);
  void setTopology2(Topology *top);
  void setTopology3(Topology *top);
  void setTopology4(Topology *top);
  void setM(int M);

  Topology** getTopology();
  Topology* getTopology1();
  Topology* getTopology2();
  Topology* getTopology3();
  Topology* getTopology4();
  int getM();
};

template <typename T>
class RnnBase {
  public:
  	 virtual void train(T *a, T *b, T alpha, T eta) = 0;
  	 virtual void feedForward(T *h_in, T *c_in,T *a, T *c_out, T *h_out) = 0;
  	// virtual void destroy() = 0;
  	// virtual T obtainError(T *b) = 0;
    //
  	// virtual void print_out() = 0;

  private:
     //virtual void prepare(rnnConfig mRnnConf) = 0;
     virtual	void init(FILE *pFile)=0;
  	// virtual void calc_feedForward() = 0;
};


class RnnSerial : public RnnBase<double> {
  private:

    int M;

    AnnSerial *ann1;
    AnnSerial *ann2;
    AnnSerial *ann3;
    AnnSerial *ann4;

    double * c_current;
    double * c_new;
    double * h_current;
    double * h_new;
    double * b;
    double * a1_output;
    double * a2_output;
    double * a3_output;
    double * a4_output;

    //
  	// int L;
  	// int * l;
  	// int * s;
  	// double * a_arr;
  	// double * z_arr;
  	// int * W;
  	// int * sw;
  	// double * w_arr;
  	// double * dw_arr;
  	// double * t_arr;
  	// double * gjl;
  public:
  	 void train(double *a, double *b, double alpha, double eta){};
  	 void feedForward(double *h_in, double *c_in,double *a, double *c_out, double *h_out);
  	// void destroy();
    //
  	// double obtainError(double *b);
  	// void print_out();
    //
    // void setWeights(double *t_w_arr){
    //   w_arr=t_w_arr;
    // };
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
    AnnSerial* getANN1();
    AnnSerial* getANN2();
    AnnSerial* getANN3();
    AnnSerial* getANN4();

    void backPropagation();

  	// double* getWeights();
    // double* getDWeights();
  	// double* getA();
    // Topology* getTopology();
    //
    // void printf_Network(string filename);

    void destroy();
  private:
    void prepare(rnnConfig *mRnnConf);
    void init(FILE *pFile);
    //
  	// void calc_feedForward();
  	// double delta_w(double grad, double dw, double alpha, double eta);
  	// double f(double x);
  	// double f_deriv(double x);
  	// double gL(double a, double z, double t);
  	// double w_gradient(int layer_id, int w_i, int w_j);
  	// void calc_gjl();
    //
    // void readf_Network(FILE *pFile);
};

double f(double x);
double f_deriv(double x);

double f_tanh(double x);
double f_tanh_deriv(double x);



#endif /* RNN_HEADER */
