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


#endif /* TESTS_HEADER */
