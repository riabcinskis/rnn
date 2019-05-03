
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

bool test_ann_feedforward_tanh(){
  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  int M = 2;

  AnnSerialDBL_tanh *serialDBL = new AnnSerialDBL_tanh(topology,M);

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

 //printf("output = %.20f\n", output[0]);

  //              0.884331422484791
  if(output[0] != 0.88433142248479079672) return false;

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

bool test_rnn_feedforward(){
  Topology *topology1 = new Topology();
  topology1->addLayer(2);
  topology1->addLayer(1);
  topology1->addLayer(2);//7weights+2

  Topology *topology2 = new Topology();
  topology2->addLayer(2);
  topology2->addLayer(2);
  topology2->addLayer(2);//12weights+4

  Topology *topology3 = new Topology();
  topology3->addLayer(2);
  topology3->addLayer(2);
  topology3->addLayer(2);//12weights+4

  Topology *topology4 = new Topology();
  topology4->addLayer(2);
  topology4->addLayer(1);
  topology4->addLayer(1);
  topology4->addLayer(2);//9weights+2

  int M = 2;

  rnnConfig *conf = new rnnConfig();
  conf->setTopology1(topology1);
  conf->setTopology2(topology2);
  conf->setTopology3(topology3);
  conf->setTopology4(topology4);
  conf->setM(M);


  RnnSerialDBL *serialDBL = new RnnSerialDBL(conf);

  double *warr1 = new double[7];
  int idx1 = 0;
  warr1[idx1++] = 0.5;
  warr1[idx1++] = 0.2;
  warr1[idx1++] = 0.0;

  warr1[idx1++] = 0.1;
  warr1[idx1++] = 0.2;
  warr1[idx1++] = 0.7;
  warr1[idx1++] = 0.9;

  double *warr2 = new double[12];
  int idx2 = 0;
  warr2[idx2++] = 0.5;
  warr2[idx2++] = 0.2;
  warr2[idx2++] = 0.0;
  warr2[idx2++] = 0.1;
  warr2[idx2++] = 0.2;
  warr2[idx2++] = 0.7;

  warr2[idx2++] = 0.9;
  warr2[idx2++] = 0.0;
  warr2[idx2++] = 0.1;
  warr2[idx2++] = 0.2;
  warr2[idx2++] = 0.7;
  warr2[idx2++] = 0.9;

  double *warr3 = new double[12];
  int idx3 = 0;
  warr3[idx3++] = 0.5;
  warr3[idx3++] = 0.2;
  warr3[idx3++] = 0.0;
  warr3[idx3++] = 0.1;
  warr3[idx3++] = 0.2;
  warr3[idx3++] = 0.7;

  warr3[idx3++] = 0.0;
  warr3[idx3++] = 0.1;
  warr3[idx3++] = 0.2;
  warr3[idx3++] = 0.7;
  warr3[idx3++] = 0.9;
  warr3[idx3++] = 0.9;

  double *warr4 = new double[9];
  int idx4 = 0;
  warr4[idx4++] = 0.5;
  warr4[idx4++] = 0.2;
  warr4[idx4++] = 0.0;

  warr4[idx4++] = 0.1;
  warr4[idx4++] = 0.2;

  warr4[idx4++] = 0.7;
  warr4[idx4++] = 0.9;
  warr4[idx4++] = 0.2;
  warr4[idx4++] = 0.7;



  double *wharr1 = new double[2];
  int idxh1 = 0;
  wharr1[idxh1++] = 0.5;
  wharr1[idxh1++] = 0.4;

  double *wharr2 = new double[4];
  int idxh2 = 0;
  wharr2[idxh2++] = 0.5;
  wharr2[idxh2++] = 0.4;
  wharr2[idxh2++] = 0.3;
  wharr2[idxh2++] = 0.1;

  double *wharr3 = new double[4];
  int idxh3 = 0;
  wharr3[idxh3++] = 0.5;
  wharr3[idxh3++] = 0.3;
  wharr3[idxh3++] = 0.3;
  wharr3[idxh3++] = 0.2;

  double *wharr4 = new double[2];
  int idxh4 = 0;
  wharr4[idxh4++] = 0.5;
  wharr4[idxh4++] = 0.1;

  serialDBL->getANN1()->setWeights(warr1, wharr1);
  serialDBL->getANN2()->setWeights(warr2, wharr2);
  serialDBL->getANN3()->setWeights(warr3, wharr3);
  serialDBL->getANN4()->setWeights(warr4, wharr4);

  //serialDBL->printf_Network("w_and_dw_tests.bin");
  double *h_input = new double[2];
  h_input[0] = 3;
  h_input[1] = 4;

  double *input = new double[2];
  input[0] = 1;
  input[1] = 2;

  double *output = new double[2];


  ///
  ///1
  ///
  double *warr_1 = serialDBL->getANN1()->getWeights();
  double *wharr_1 = serialDBL->getANN1()->getHWeights();

  for(int i = 0; i < 7; i++){
  //   printf("w[%d] = %.20f\n", i, warr_1[i]);
     if(warr_1[i]!=warr1[i]) return false;
  }

  for(int i = 0; i < 2; i++){
  //  printf("wh[%d] = %.20f\n", i, wharr2[i]);
    if(wharr1[i]!=wharr_1[i]) return false;
  }

	serialDBL->getANN1()->feedForward(h_input,input, output);
  // printf("output = %.20f\n", output[0]);
  // printf("output = %.20f\n", output[1]);


  //              0.689589607251556
  if(output[0] != 0.68958960725155571403) return false;
  //              0.74958548419844700000
  if(output[1] != 0.74958548419844750477) return false;

  if(serialDBL->getANN1()->getA()[2] != 1) return false;
  if(serialDBL->getANN1()->getA()[4] != 1) return false;


  ///
  ///2
  ///
  double *warr_2 = serialDBL->getANN2()->getWeights();
  double *wharr_2 = serialDBL->getANN2()->getHWeights();

  for(int i = 0; i < 12; i++){
  //   printf("w[%d] = %.20f\n", i, warr_2[i]);
     if(warr_2[i]!=warr2[i]) return false;
 }

  for(int i = 0; i < 4; i++){
  //  printf("wh[%d] = %.20f\n", i, wharr2[i]);
    if(wharr2[i]!=wharr_2[i]) return false;
  }

	serialDBL->getANN2()->feedForward(h_input,input, output);
  //printf("output = %.20f\n", output[0]);
  // printf("output = %.20f\n", output[1]);


  //              0.84085944955203300000
  if(output[0] != 0.84085944955203295592) return false;
  //              0.74789281328782
  if(output[1] != 0.74789281328782009073) return false;

  if(serialDBL->getANN2()->getA()[2] != 1) return false;
  if(serialDBL->getANN2()->getA()[5] != 1) return false;

  ///
  ///3
  ///
  double *warr_3 = serialDBL->getANN3()->getWeights();
  double *wharr_3 = serialDBL->getANN3()->getHWeights();

  for(int i = 0; i < 12; i++){
  //   printf("w[%d] = %.20f\n", i, warr_1[i]);
     if(warr_3[i]!=warr3[i]) return false;
  }

  for(int i = 0; i < 4; i++){
  //  printf("wh[%d] = %.20f\n", i, wharr2[i]);
    if(wharr3[i]!=wharr_3[i]) return false;
  }

	serialDBL->getANN3()->feedForward(h_input,input, output);
  //printf("output = %.20f\n", output[0]);
  //printf("output = %.20f\n", output[1]);


  //              0.79996904340460500000
  if(output[0] != 0.79996904340460495142) return false;
  //              0.934733066362687
  if(output[1] != 0.93473306636268660430) return false;

  if(serialDBL->getANN3()->getA()[2] != 1) return false;
  if(serialDBL->getANN3()->getA()[5] != 1) return false;

  ///
  ///4
  ///
  double *warr_4 = serialDBL->getANN4()->getWeights();
  double *wharr_4 = serialDBL->getANN4()->getHWeights();

  for(int i = 0; i < 9; i++){
  //   printf("w[%d] = %.20f\n", i, warr_1[i]);
     if(warr_4[i]!=warr4[i]) return false;
 }

  for(int i = 0; i < 2; i++){
  //  printf("wh[%d] = %.20f\n", i, wharr2[i]);
    if(wharr4[i]!=wharr_4[i]) return false;
  }

	serialDBL->getANN4()->feedForward(h_input,input, output);
  //printf("output = %.20f\n", output[0]);
  //printf("output = %.20f\n", output[1]);


  //              0.645914426127551
  if(output[0] != 0.64591442612755145536) return false;
  //              0.771312387165297
  if(output[1] != 0.77131238716529726407) return false;

  if(serialDBL->getANN4()->getA()[2] != 1) return false;
  if(serialDBL->getANN4()->getA()[4] != 1) return false;


  delete [] warr1;
  delete [] wharr1;
  delete [] wharr_1;
  delete [] warr_1;

  delete [] warr2;
  delete [] wharr2;
  delete [] wharr_2;
  delete [] warr_2;

  delete [] warr3;
  delete [] wharr3;
  delete [] wharr_3;
  delete [] warr_3;

  delete [] warr4;
  delete [] wharr4;
  delete [] wharr_4;
  delete [] warr_4;

  delete [] h_input;
  delete [] input;
  delete [] output;
  delete serialDBL;
  delete topology1;
  delete topology2;
  delete topology3;
  delete topology4;

  return true;
}

bool run_tests(){

  printf("running tests ... \n");

  int failCount = 0;

  bool passed = test_topology(); failCount += passed ? 0 : 1;
  printf("%s - test_topology\n", passed ? "PASSED" : "FAILED");

  passed = test_ann_feedforward(); failCount += passed ? 0 : 1;
  printf("%s - test_ann_feedforward\n", passed ? "PASSED" : "FAILED");

  passed = test_ann_feedforward_tanh(); failCount += passed ? 0 : 1;
  printf("%s - test_ann_feedforward_tanh\n", passed ? "PASSED" : "FAILED");

  passed = test_rnn_feedforward(); failCount += passed ? 0 : 1;
  printf("%s - test_rnn_feedforwards_of_networks\n", passed ? "PASSED" : "FAILED");


  printf("\n");
  if(failCount == 0) printf("ALL tests PASSED\n");
  else printf("%d TESTS FAILED\n", failCount);

  return failCount == 0;
}
