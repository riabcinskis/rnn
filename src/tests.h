#ifndef TESTS_HEADER
#define TESTS_HEADER

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
    int u;
    int V;
    int M;
    Topology **cTopology;

    int **vsW;
    int **vl;
    int *vL;

    Derivatives ** deriv_in;
    Derivatives ** deriv_out;

  public:
    DerivIO(Topology **top, int M, int V, int u);

    ~DerivIO();

    void setDeriv(int v,int s,int i, int j, int k, double deriv);
    void setHDeriv(int v, int i, int j, int k, double deriv);

    Derivatives** getDeriv_in();
    Derivatives** getDeriv_out();

    double getDerivInValue(int v,int s,int i, int j, int k);
    double getDerivInValueH(int v,int i, int j, int k);
    double getDerivOutValue(int v,int s,int i, int j, int k);
    double getDerivOutValueH(int v,int i, int j, int k);


    int deriv(int l, int i, int j);

    int hderiv(int i, int j);

  private:
    int vi(int v, int s, int i, int j, int k);
    int vhi(int v, int i, int j, int k);
};


#endif /* TESTS_HEADER */
