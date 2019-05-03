#ifndef RNN_HEADER
#define RNN_HEADER

//#include <helper_cuda.h>
#include "ann.h"

#include <cmath>


/* Class definitions here. */
void run_cuda_sample();

class rnnConfig{
private:
  Topology* cTopology1;
  Topology* cTopology2;
  Topology* cTopology3;
  Topology* cTopology4;
  int mM;

public:
  void setTopology1(Topology *top);
  void setTopology2(Topology *top);
  void setTopology3(Topology *top);
  void setTopology4(Topology *top);
  void setM(int M);

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
  	// virtual void feedForward(T *a, T *b) = 0;
  	// virtual void destroy() = 0;
  	// virtual T obtainError(T *b) = 0;
    //
  	// virtual void print_out() = 0;

  private:
     //virtual void prepare(rnnConfig mRnnConf) = 0;
     virtual	void init(FILE *pFile)=0;
  	// virtual void calc_feedForward() = 0;
};


class RnnSerialDBL : public RnnBase<double> {
  private:

    int M;

    AnnSerialDBL *ann1;
    AnnSerialDBL *ann2;
    AnnSerialDBL_tanh *ann3;
    AnnSerialDBL *ann4;
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
  	// void feedForward(double *a, double *b);
  	// void destroy();
    //
  	// double obtainError(double *b);
  	// void print_out();
    //
    // void setWeights(double *t_w_arr){
    //   w_arr=t_w_arr;
    // };
    RnnSerialDBL(rnnConfig *mRnnConf) {
      prepare(mRnnConf);
      init(NULL);
    };

  	RnnSerialDBL(string filename) {
        // FILE * p1File;
        // p1File = fopen(filename.c_str(), "rb");
        // Topology *top=new Topology();
        // top->readTopology(p1File);
        // prepare(top);
        // init(p1File);
        // fclose (p1File);
    };

    //
    AnnSerialDBL* getANN1();
    AnnSerialDBL* getANN2();
    AnnSerialDBL_tanh* getANN3();
    AnnSerialDBL* getANN4();

  	// double* getWeights();
    // double* getDWeights();
  	// double* getA();
    // Topology* getTopology();
    //
    // void printf_Network(string filename);


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





#endif /* RNN_HEADER */
