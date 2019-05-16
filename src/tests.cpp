
#include <stdio.h>
#include "rnn.h"

bool test_topology(){

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  if(topology->obtainNeuronCount() != 7) return false;
  if(topology->obtainWeightCount() != 9) return false;

  delete topology;
  return true;
}

bool test_ann_feedforward(){

  Topology **topology = new Topology*[1];
  topology[0] = new Topology();
  topology[0]->addLayer(2);
  topology[0]->addLayer(2);
  topology[0]->addLayer(2);

  int M = 2;
  //printf("%.20f\n", f(2.0));
  double (*func)(double);
  double (*func_deriv)(double);
  func=f;
  func_deriv = f_deriv;

  AnnSerial *serialDBL = new AnnSerial(1,0,M,topology,func,func_deriv);

  double *warr = new double[12];
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

  double *output = new double[2];

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

  // printf("output = %.20f\n", output[1]);
  //              0.795489675867213
  if(output[0] != 0.79548967586721286427) return false;

  //              0.791441326894792
  if(output[1] != 0.79144132689479196330) return false;

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
  Topology **topology = new Topology*[1];
  topology[0] = new Topology();
  topology[0]->addLayer(2);
  topology[0]->addLayer(2);
  topology[0]->addLayer(2);

  int M = 2;

  double (*func)(double);
  double (*func_deriv)(double);
  func=f_tanh;
  func_deriv = f_tanh_deriv;

  AnnSerial *serialDBL = new AnnSerial(1,0,M,topology,func,func_deriv);

  double *warr = new double[12];
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

  //              0.884527266372134
  if(output[0] != 0.88452726637213452410) return false;
  //              0.883443223178355
  if(output[1] != 0.88344322317835477509) return false;

  if(serialDBL->getA()[2] != 1) return false;
  if(serialDBL->getA()[5] != 1) return false;


 //  printf("A3:  %f\n", serialDBL->getA()[3]);

  delete [] warr;
  delete [] wharr;
  delete [] input;
  delete [] output;
  delete serialDBL;
  delete [] topology;

  return true;
}

bool test_rnn_feedforward(){

  Topology **topology = new Topology*[4];
  topology[0] = new Topology();
  topology[0]->addLayer(2);
  topology[0]->addLayer(1);
  topology[0]->addLayer(2);//7weights+2

  topology[1] = new Topology();
  topology[1]->addLayer(2);
  topology[1]->addLayer(2);
  topology[1]->addLayer(2);//12weights+4

  topology[2] = new Topology();
  topology[2]->addLayer(2);
  topology[2]->addLayer(2);
  topology[2]->addLayer(2);//12weights+4

  topology[3] = new Topology();
  topology[3]->addLayer(2);
  topology[3]->addLayer(1);
  topology[3]->addLayer(1);
  topology[3]->addLayer(2);//9weights+2

  int M = 2;

  rnnConfig *conf = new rnnConfig();
  conf->setTopology(topology);
  // conf->setTopology2(topology2);
  // conf->setTopology3(topology3);
  // conf->setTopology4(topology4);
  conf->setM(M);

  RnnSerial *serialDBL = new RnnSerial(conf);

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

  serialDBL->destroy();

//   delete [] warr1;
//   delete [] wharr1;
//   delete [] wharr_1;
//   delete [] warr_1;

//   delete [] warr2;
//   delete [] wharr2;
//   delete [] wharr_2;
//   delete [] warr_2;
//
//   delete [] warr3;
//   delete [] wharr3;
//   delete [] wharr_3;
//   delete [] warr_3;
//
//   delete [] warr4;
//   delete [] wharr4;
//   delete [] wharr_4;
//   delete [] warr_4;

  delete [] h_input;
  delete [] input;
  delete [] output;
  delete [] topology;
  delete conf;
  delete serialDBL;
  return true;
}

bool test_rnn_feedforward_full(){
  Topology **topology = new Topology*[4];
  topology[0] = new Topology();
  topology[0]->addLayer(2);
  topology[0]->addLayer(1);
  topology[0]->addLayer(2);//7weights+2

  topology[1] = new Topology();
  topology[1]->addLayer(2);
  topology[1]->addLayer(2);
  topology[1]->addLayer(2);//12weights+4

  topology[2] = new Topology();
  topology[2]->addLayer(2);
  topology[2]->addLayer(2);
  topology[2]->addLayer(2);//12weights+4

  topology[3] = new Topology();
  topology[3]->addLayer(2);
  topology[3]->addLayer(1);
  topology[3]->addLayer(1);
  topology[3]->addLayer(2);//9weights+2


  int M = 2;

  rnnConfig *conf = new rnnConfig();
  conf->setTopology(topology);
  // conf->setTopology2(topology2);
  // conf->setTopology3(topology3);
  // conf->setTopology4(topology4);
  conf->setM(M);


  RnnSerial *serialDBL = new RnnSerial(conf);

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

  double *output1 = new double[2];
  double *output2 = new double[2];
  double *output3 = new double[2];
  double *output4 = new double[2];


  ///
  ///1
  ///
  	serialDBL->getANN1()->feedForward(h_input,input, output1);
  //              0.689589607251556
  if(output1[0] != 0.68958960725155571403) return false;
  //              0.74958548419844700000
  if(output1[1] != 0.74958548419844750477) return false;

  ///
  ///2
  ///
	serialDBL->getANN2()->feedForward(h_input,input, output2);
  //              0.84085944955203300000
  if(output2[0] != 0.84085944955203295592) return false;
  //              0.74789281328782
  if(output2[1] != 0.74789281328782009073) return false;

  ///
  ///3
  ///
	serialDBL->getANN3()->feedForward(h_input,input, output3);


  //              0.79996904340460500000
  if(output3[0] != 0.79996904340460495142) return false;
  //              0.934733066362687
  if(output3[1] != 0.93473306636268660430) return false;

  ///
  ///4
  ///
	serialDBL->getANN4()->feedForward(h_input,input, output4);

  //              0.645914426127551
  if(output4[0] != 0.64591442612755145536) return false;
  //              0.771312387165297
  if(output4[1] != 0.77131238716529726407) return false;

  double *c_in = new double[2];
  c_in[0] = 0.5;
  c_in[1] = 0.3;
  double *c_out = new double[2];
  double *h_out = new double[2];
  serialDBL->feedForward(h_input,c_in,input,c_out, h_out);

  //             1.01745633312164000000
  if(c_out[0] != 1.01745633312164018847) return false;
  //             0.923955787934675
  if(c_out[1] != 0.92395578793467447731) return false;
  //             0.469404624141351
  if(h_out[0] != 0.46940462414135097902) return false;
  //             0.530595375858649
  if(h_out[1] != 0.53059537585864913201) return false;

  c_in[0] = c_out[0];
  c_in[1] = c_out[1];
  h_input[0] = h_out[0];
  h_input[1] = h_out[1];

  serialDBL->feedForward(h_input,c_in,input,c_out, h_out);

  //             1.33881207992600000000
  if(c_out[0] != 1.33881207992600104184) return false;
  //             1.3692467644635
  if(c_out[1] != 1.36924676446350157555) return false;
  //             0.453692800787076
  if(h_out[0] != 0.45369280078707552306) return false;
  //             0.546307199212925
  if(h_out[1] != 0.54630719921292447694) return false;


  serialDBL->destroy();
  delete [] warr1;
  delete [] wharr1;

  delete [] warr2;
  delete [] wharr2;

  delete [] warr3;
  delete [] wharr3;

  delete [] warr4;
  delete [] wharr4;

  delete [] h_input;
  delete [] input;

  delete [] output1;
  delete [] output2;
  delete [] output3;
  delete [] output4;
  delete [] topology;
  delete conf;
  delete serialDBL;
  return true;
}



bool run_tests(){

  printf("running tests ... \n");

  int failCount = 0;

  bool passed = test_topology(); failCount += passed ? 0 : 1;
  printf("%s - test_topology\n", passed ? "PASSED" : "FAILED");
  printf("%s\n", "---------------------");
  passed = test_ann_feedforward(); failCount += passed ? 0 : 1;
  printf("%s - test_ann_feedforward\n", passed ? "PASSED" : "FAILED");
  printf("%s\n", "---------------------");
  passed = test_ann_feedforward_tanh(); failCount += passed ? 0 : 1;
  printf("%s - test_ann_feedforward_tanh\n", passed ? "PASSED" : "FAILED");
  printf("%s\n", "---------------------");
  passed = test_rnn_feedforward(); failCount += passed ? 0 : 1;
  printf("%s - test_rnn_feedforwards_of_networks\n", passed ? "PASSED" : "FAILED");
  printf("%s\n", "---------------------");
  passed = test_rnn_feedforward_full(); failCount += passed ? 0 : 1;
  printf("%s - test_rnn_feedforwards_full\n", passed ? "PASSED" : "FAILED");


  printf("\n");
  if(failCount == 0) printf("ALL tests PASSED\n");
  else printf("%d TESTS FAILED\n", failCount);

  return failCount == 0;
}
