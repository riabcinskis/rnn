
#include <stdio.h>
#include "rnn.h"

bool test_topology(){

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  if(topology->obtainNeuronCount() != 8) return false;
  if(topology->obtainWeightCount() != 9) return false;

  return true;
}

bool test_ann_feedforward(){
  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  int M = 2;

  AnnSerialDBL *serialDBL = new AnnSerialDBL(topology,M);

  double *warr = new double[9];
  int idx = 0;
  warr[idx++] = 0.5;
  warr[idx++] = 0.2;
  warr[idx++] = 0.0;

  warr[idx++] = 0.1;
  warr[idx++] = 0.2;
  warr[idx++] = 0.7;

  warr[idx++] = 0.9;
  warr[idx++] = 0.3;
  warr[idx++] = 0.2;

  double *wharr = new double[4];
  int idxh = 0;
  wharr[idxh++] = 0.5;
  wharr[idxh++] = 0.4;
  wharr[idxh++] = 0.3;
  wharr[idxh++] = 0.1;

  serialDBL->setWeights(warr, wharr);

  //serialDBL->printf_Network("w_and_dw_tests.bin");
  double *h_input = new double[2];
  h_input[0] = 3;
  h_input[1] = 4;

  double *input = new double[2];
  input[0] = 1;
  input[1] = 2;

  double *output = new double[1];

  double *warr2 = serialDBL->getWeights();
  double *wharr2 = serialDBL->getHWeights();

  for(int i = 0; i < 9; i++){
     //printf("w[%d] = %.20f\n", i, warr2[i]);
     if(warr2[i]!=warr[i]) return false;
 }

  for(int i = 0; i < 4; i++){
  //  printf("wh[%d] = %.20f\n", i, wharr2[i]);
    if(wharr2[i]!=wharr[i]) return false;
  }

	serialDBL->feedForward(h_input,input, output);

// printf("output = %.20f\n", output[0]);

  //                0.794463281942811
  if(output[0] != 0.79446328194281123913) return false;

  if(serialDBL->getA()[2] != 1) return false;
  if(serialDBL->getA()[5] != 1) return false;

//  printf("A3:  %f\n", serialDBL->getA()[3]);

  delete [] warr;
  delete [] wharr;
  delete [] input;
  delete [] output;
  delete serialDBL;
  delete topology;

  return true;
}


bool run_tests(){

  printf("running tests ... \n");

  int failCount = 0;

  bool passed = test_topology(); failCount += passed ? 0 : 1;
  printf("%s - test_topology\n", passed ? "PASSED" : "FAILED");

  passed = test_ann_feedforward(); failCount += passed ? 0 : 1;
  printf("%s - test_ann_feedforward\n", passed ? "PASSED" : "FAILED");


  printf("\n");
  if(failCount == 0) printf("ALL tests PASSED\n");
  else printf("%d TESTS FAILED\n", failCount);

  return failCount == 0;
}
