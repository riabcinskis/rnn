#ifndef TESTS_HEADER
#define TESTS_HEADER


#include <stdio.h>
#include "rnn.h"
#include "app.h"
#include <cmath>
#include <ctime>


struct Derivatives;

bool run_tests();

class WeightIO {
  private:
    int M;
    Topology *cTopology;
    int *sW;

    double *w;
    double *wh;

  public:
    WeightIO(Topology *top, int M);

    ~WeightIO();

    void setHWeight(int i, int j, double weight);
    void setWeight(int l, int i, int j, double weight);

    double* getWeights();
    double* getHWeights();

  private:
    int wij(int l, int i, int j);
    int whij(int i, int j);

};

class DerivIO {
  private:
    int M;
    Topology *cTopology;

    int *sW;
    int *l;
    int L;

    Derivatives *deriv;

  public:
    DerivIO(Topology *top, int M);

    ~DerivIO();

    void setDeriv(int s, int i, int j, int k, double deriv);
    void setHDeriv(int i, int j, int k, double deriv);

    Derivatives* getDerivatives();

    double getDeriv(int s, int i, int j, int k);
    double getHDeriv(int i, int j, int k);

  private:
    int vi(int s, int i, int j, int k);
    int vhi(int i, int j, int k);
};


#endif /* TESTS_HEADER */
