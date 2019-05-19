
#include "tests.h"


//
// WeightIO
//
WeightIO::WeightIO(Topology *top, int M){
  this->cTopology = top;
  this->M = M;

  int L = top->getLayerCount();

  assert(L > 1);
  assert(M == top->getLayerSize(L-1));

  sW = new int[top->getLayerCount()-1];

  w = new double[top->obtainWeightCount()];
  wh = new double[M*top->getLayerSize(1)];

  obtainSW(top, sW);
}

WeightIO::~WeightIO(){
  delete [] sW;
}

void WeightIO::setHWeight(int i, int j, double weight){
  assert(i >= 0);
  assert(j >= 0);
  assert(i < M);
  assert(j < cTopology->getLayerSize(1));

  wh[whij(i, j)] = weight;
}

void WeightIO::setWeight(int l, int i, int j, double weight){

  assert(l >= 0);
  assert(l < cTopology->getLayerCount());

  assert(i >= 0);
  assert(j >= 0);
  assert(i < cTopology->getLayerSize(l)+1);
  assert(j < cTopology->getLayerSize(l+1));

  w[wij(l, i, j)] = weight;
}

double* WeightIO::getWeights(){
  return w;
}

double* WeightIO::getHWeights(){
  return wh;
}

int WeightIO::wij(int l, int i, int j){
  return sW[l] + i*cTopology->getLayerSize(l+1) + j;
}

int WeightIO::whij(int i, int j){
  return i*cTopology->getLayerSize(1) + j;
}

//
//DerivIO
//

DerivIO::DerivIO(Topology *top, int M){
  this->cTopology = top;
  this->M = M;

  sW = new int[top->getLayerCount()-1];

  obtainSW(top, sW);

  assert(sW[0] == 0);

  deriv = new Derivatives;
  deriv->v = new double[top->obtainWeightCount()*M];
  deriv->vh = new double[M*top->getLayerSize(1)*M];

  // printf("deriv->vh[0] =%f \n", deriv->vh[0]);


  int L = top->getLayerCount();

  assert(L > 1);
  assert(M == top->getLayerSize(L-1));
}

DerivIO::~DerivIO(){
  delete [] sW;
}

void DerivIO::setHDeriv(int i, int j, int k, double deriv){
  assert(i >= 0);
  assert(j >= 0);
  assert(k >= 0);

  assert(i < M);
  assert(j < cTopology->getLayerSize(1));
  assert(k < M);

  this->deriv->vh[vhi(i, j, k)] = deriv;
}

void DerivIO::setDeriv(int s, int i, int j, int k, double deriv){

  assert(i >= 0);
  assert(j >= 0);
  assert(k >= 0);
  assert(s >= 0);
  assert(s < cTopology->getLayerCount()-1);
  assert(i < cTopology->getLayerSize(s)+1);
  assert(j < cTopology->getLayerSize(s+1));
  assert(k < M);

  this->deriv->v[vi(s, i, j, k)] = deriv;
}

Derivatives* DerivIO::getDerivatives(){
  return deriv;
}

double DerivIO::getDeriv(int s, int i, int j, int k){
  return deriv->v[vi(s, i, j, k)];
}

double DerivIO::getHDeriv(int i, int j, int k){
  return deriv->vh[vhi(i, j, k)];
}

int DerivIO::vi(int s, int i, int j, int k){
  return (sW[s] + i*(cTopology->getLayerSize(s+1)) + j)*M + k;
}

int DerivIO::vhi(int i, int j, int k){
  return  (i*(cTopology->getLayerSize(1)) + j)*M + k;
}


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

  int M = 2;

  Topology **topology = new Topology*[1];
  topology[0] = new Topology();
  topology[0]->addLayer(2);
  topology[0]->addLayer(2);
  topology[0]->addLayer(M);

  //printf("%.20f\n", f(2.0));
  double (*func)(double);
  double (*func_deriv)(double);
  func = f;
  func_deriv = f_deriv;


  WeightIO* weightIO = new WeightIO(topology[0], M);

  // double *warr = new double[12];
  // int idx = 0;
  weightIO->setWeight(0, 0, 0, 0.5); // warr[idx++] = 0.5;
  weightIO->setWeight(0, 0, 1, 0.2); // warr[idx++] = 0.2;
  //
  weightIO->setWeight(0, 1, 0, 0.0); // warr[idx++] = 0.0;
  weightIO->setWeight(0, 1, 1, 0.1); // warr[idx++] = 0.1;
  //
  weightIO->setWeight(0, 2, 0, 0.2); // warr[idx++] = 0.2;
  weightIO->setWeight(0, 2, 1, 0.7); // warr[idx++] = 0.7;
  //
  //
  weightIO->setWeight(1, 0, 0, 0.9); // warr[idx++] = 0.9;
  weightIO->setWeight(1, 0, 1, 0.3); // warr[idx++] = 0.3;
  //
  weightIO->setWeight(1, 1, 0, 0.2); // warr[idx++] = 0.2;
  weightIO->setWeight(1, 1, 1, 0.9); // warr[idx++] = 0.9;
  //
  weightIO->setWeight(1, 2, 0, 0.3); // warr[idx++] = 0.3;
  weightIO->setWeight(1, 2, 1, 0.2); // warr[idx++] = 0.2;
  //
  // double *wharr = new double[4];
  // int idxh = 0;
  weightIO->setHWeight(0, 0, 0.5); // wharr[idxh++] = 0.5;
  weightIO->setHWeight(0, 1, 0.4); // wharr[idxh++] = 0.4;
  //
  weightIO->setHWeight(1, 0, 0.3); // wharr[idxh++] = 0.3;
  weightIO->setHWeight(1, 1, 0.1); // wharr[idxh++] = 0.1;

  AnnSerial *serialDBL = new AnnSerial(1, 0, M, topology, func, func_deriv);
  serialDBL->setWeights(weightIO->getWeights(), weightIO->getHWeights());

  //serialDBL->printf_Network("w_and_dw_tests.bin");
  double *h_input = new double[2];
  h_input[0] = 3;
  h_input[1] = 4;

  double *input = new double[2];
  input[0] = 1;
  input[1] = 2;

  double *output = new double[M];

  double *warr2 = serialDBL->getWeights();
  double *wharr2 = serialDBL->getHWeights();

  for(int i = 0; i < 9; i++){
     //printf("w[%d] = %.20f\n", i, warr2[i]);
     if(warr2[i]!=weightIO->getWeights()[i]) return false;
 }


  for(int i = 0; i < 4; i++){
  //  printf("wh[%d] = %.20f\n", i, wharr2[i]);
    if(wharr2[i]!=weightIO->getHWeights()[i]) return false;
  }

	serialDBL->feedForward(h_input,input, output);

  // printf("output = %.20f\n", output[0]);
  //              0.795489675867213
  if(output[0] != 0.79548967586721286427) {printf("fail : 0, \n");return false;}

  //              0.791441326894792
  if(output[1] != 0.79144132689479196330) {printf("fail : 1\n");return false;}

  if(serialDBL->getA()[2] != 1) return false;
  if(serialDBL->getA()[5] != 1) return false;

  //  printf("A3:  %f\n", serialDBL->getA()[3]);
  serialDBL->destroy();
  delete weightIO;
  delete [] input;
  delete [] output;
  delete serialDBL;
  delete topology;

  return true;
}


bool test_backprogg(){

  Topology **topology = new Topology*[2];
  topology[0] = new Topology();
  topology[0]->addLayer(2);
  topology[0]->addLayer(2);//

  topology[1] = new Topology();
  topology[1]->addLayer(2);
  topology[1]->addLayer(2);//

  int I = 2; // number of inputs
  int M = 2; // number of outputs
  int V = 2;

  WeightIO* weightIO = new WeightIO(topology[0], M);

  weightIO->setWeight(0, 0, 0, 0.1); // warr[idx++] = 0.5;
  weightIO->setWeight(0, 0, 1, 0.2); // warr[idx++] = 0.2;

  weightIO->setWeight(0, 1, 0, 0.3); // warr[idx++] = 0.5;
  weightIO->setWeight(0, 1, 1, 0.4); // warr[idx++] = 0.2;

  weightIO->setWeight(0, 2, 0, 0.5); // warr[idx++] = 0.5;
  weightIO->setWeight(0, 2, 1, 0.6); // warr[idx++] = 0.2;

  weightIO->setHWeight(0, 0, 0.3); // wharr[idxh++] = 0.3;
  weightIO->setHWeight(0, 1, 0.1); // wharr[idxh++] = 0.1;

  weightIO->setHWeight(1, 0, 0.3); // wharr[idxh++] = 0.3;
  weightIO->setHWeight(1, 1, 0.1); // wharr[idxh++] = 0.1;


  AnnSerial *ann = new AnnSerial(V, 0, M, topology, f, f_deriv);
  ann->setWeights(weightIO->getWeights(), weightIO->getHWeights());



  double *h_input = new double[M];
  h_input[0] = 3;
  h_input[1] = 4;

  double *input = new double[I];
  input[0] = 1;
  input[1] = 2;

  double *output = new double[M];


  ann->feedForward(h_input, input, output);
  // printf("output = %.20f\n", output[0]);
   //printf("output = %.20f\n", output[1]);


  //              0.964428810727364
  if(output[0] != 0.96442881072736386106) return false;
  //              0.908877038985144
  if(output[1] != 0.90887703898514382583) return false;

  if(ann->getA()[2] != 1) return false;



  DerivIO* derivIO_in0 = new DerivIO(topology[0], M);
  DerivIO* derivIO_in1 = new DerivIO(topology[1], M);

  DerivIO* derivIO_out0 = new DerivIO(topology[0], M);
  DerivIO* derivIO_out1 = new DerivIO(topology[1], M);

  ///                   s  i  j  k  val
  derivIO_in0->setDeriv(0, 0, 0, 0, -0.05);
  derivIO_in0->setDeriv(0, 0, 0, 1, -0.15);

  derivIO_in0->setDeriv(0, 0, 1, 0, 0.1);
  derivIO_in0->setDeriv(0, 0, 1, 1, 0.03);

  derivIO_in0->setDeriv(0, 1, 0, 0, -0.19);
  derivIO_in0->setDeriv(0, 1, 0, 1, 0.31);

  derivIO_in0->setDeriv(0, 1, 1, 0, 0.17);
  derivIO_in0->setDeriv(0, 1, 1, 1, 0.4);

  derivIO_in0->setDeriv(0, 2, 0, 0, -0.7);
  derivIO_in0->setDeriv(0, 2, 0, 1, 0.1);

  derivIO_in0->setDeriv(0, 2, 1, 0, 0.23);
  derivIO_in0->setDeriv(0, 2, 1, 1, 0.39);


  derivIO_in0->setHDeriv(0, 0, 0, 0.71);
  derivIO_in0->setHDeriv(0, 0, 1, 0.35);

  derivIO_in0->setHDeriv(0, 1, 0, 0.21);
  derivIO_in0->setHDeriv(0, 1, 1, -0.11);

  derivIO_in0->setHDeriv(1, 0, 0, 0.2);
  derivIO_in0->setHDeriv(1, 0, 1, -0.63);

  derivIO_in0->setHDeriv(1, 1, 0, 0.1);
  derivIO_in0->setHDeriv(1, 1, 1, 0.01);








  derivIO_in1->setDeriv(0, 0, 0, 0, 0.05);
  derivIO_in1->setDeriv(0, 0, 0, 1, 0.15);

  derivIO_in1->setDeriv(0, 0, 1, 0, 0.5);
  derivIO_in1->setDeriv(0, 0, 1, 1, 0.13);

  derivIO_in1->setDeriv(0, 1, 0, 0, -0.18);
  derivIO_in1->setDeriv(0, 1, 0, 1, -0.31);

  derivIO_in1->setDeriv(0, 1, 1, 0, 0.32);
  derivIO_in1->setDeriv(0, 1, 1, 1, 0.02);

  derivIO_in1->setDeriv(0, 2, 0, 0, -0.22);
  derivIO_in1->setDeriv(0, 2, 0, 1, 0.14);

  derivIO_in1->setDeriv(0, 2, 1, 0, 0.43);
  derivIO_in1->setDeriv(0, 2, 1, 1, 0.34);


  derivIO_in1->setHDeriv(0, 0, 0, -0.71);
  derivIO_in1->setHDeriv(0, 0, 1, 0.53);

  derivIO_in1->setHDeriv(0, 1, 0, 0.35);
  derivIO_in1->setHDeriv(0, 1, 1, -0.35);

  derivIO_in1->setHDeriv(1, 0, 0, 0.2);
  derivIO_in1->setHDeriv(1, 0, 1, -0.24);

  derivIO_in1->setHDeriv(1, 1, 0, 0.22);
  derivIO_in1->setHDeriv(1, 1, 1, 0.01);



  Derivatives **deriv_in = new Derivatives*[2];
  Derivatives **deriv_out = new Derivatives*[2];

  deriv_in[0] =  derivIO_in0->getDerivatives();
  deriv_in[1] =  derivIO_in1->getDerivatives();

  deriv_out[0] =  derivIO_out0->getDerivatives();
  deriv_out[1] =  derivIO_out1->getDerivatives();


  ann->backPropagation( deriv_in, deriv_out);

  if(derivIO_out0->getDeriv(0, 0, 0, 0) != 0.03224752698038448745) return false;
  if(derivIO_out0->getDeriv(0, 0, 0, 1) != 0.08116317565092628916) return false;

  if(derivIO_out0->getDeriv(0, 0, 1, 0) != 0.03564380907725477055) return false;
  if(derivIO_out0->getDeriv(0, 0, 1, 1) != 0.08389622136162074029) return false;

  if(derivIO_out0->getDeriv(0, 1, 0, 0) != 0.06984677120432215836) return false;
  if(derivIO_out0->getDeriv(0, 1, 0, 1) != 0.16663296878537112167) return false;

  if(derivIO_out0->getDeriv(0, 1, 1, 0) != 0.07447806497278162963) return false;
  if(derivIO_out0->getDeriv(0, 1, 1, 1) != 0.17035984929995445780) return false;

  if(derivIO_out0->getDeriv(0, 2, 0, 0) != 0.02813082140842051646) return false;
  if(derivIO_out0->getDeriv(0, 2, 0, 1) != 0.07785039297129664471) return false;

  if(derivIO_out0->getDeriv(0, 2, 1, 0) != 0.04068677340291064481) return false;
  if(derivIO_out0->getDeriv(0, 2, 1, 1) != 0.08795438014416706585) return false;


  if(derivIO_out0->getHDeriv(0, 0, 0) != 0.11382690906480398552) return false;
  if(derivIO_out0->getHDeriv(0, 0, 1) != 0.25723757507324190863) return false;

  if(derivIO_out0->getHDeriv(0, 1, 0) != 0.10394681569209043848) return false;
  if(derivIO_out0->getHDeriv(0, 1, 1) != 0.24928689664213071753) return false;

  if(derivIO_out0->getHDeriv(1, 0, 0) != 0.13279806057560464283) return false;
  if(derivIO_out0->getHDeriv(1, 0, 1) != 0.32771702658236256944) return false;

  if(derivIO_out0->getHDeriv(1, 1, 0) != 0.13835561309775604166) return false;
  if(derivIO_out0->getHDeriv(1, 1, 1) != 0.33218928319986262832) return false;








  if(derivIO_out1->getDeriv(0, 0, 0, 0) != 0.00205835278598198897) return false;
  if(derivIO_out1->getDeriv(0, 0, 0, 1) != 0.00165639133981482223) return false;

  if(derivIO_out1->getDeriv(0, 0, 1, 0) != 0.00648381127584326481) return false;
  if(derivIO_out1->getDeriv(0, 0, 1, 1) != 0.00521763272041668967) return false;

  if(derivIO_out1->getDeriv(0, 1, 0, 0) != -0.00504296432565587253) return false;
  if(derivIO_out1->getDeriv(0, 1, 0, 1) != -0.00405815878254631428) return false;

  if(derivIO_out1->getDeriv(0, 1, 1, 0) != 0.00349919973616938124) return false;
  if(derivIO_out1->getDeriv(0, 1, 1, 1) != 0.00281586527768519805) return false;

  if(derivIO_out1->getDeriv(0, 2, 0, 0) != -0.00082334111439279559) return false;
  if(derivIO_out1->getDeriv(0, 2, 0, 1) != -0.00066255653592592887) return false;

  if(derivIO_out1->getDeriv(0, 2, 1, 0) != 0.00792465822603065709) return false;
  if(derivIO_out1->getDeriv(0, 2, 1, 1) != 0.00637710665828706679) return false;


  if(derivIO_out1->getHDeriv(0, 0, 0) != -0.00185251750738378985) return false;
  if(derivIO_out1->getHDeriv(0, 0, 1) != -0.00149075220583333901) return false;

  if(derivIO_out1->getHDeriv(0, 1, 0) != 0.0) return false;
  if(derivIO_out1->getHDeriv(0, 1, 1) != 0.0) return false;

  if(derivIO_out1->getHDeriv(1, 0, 0) != -0.00041167055719639768) return false;
  if(derivIO_out1->getHDeriv(1, 0, 1) != -0.00033127826796296416) return false;

  if(derivIO_out1->getHDeriv(1, 1, 0) != 0.00236710570387928731) return false;
  if(derivIO_out1->getHDeriv(1, 1, 1) != 0.00190485004078704587) return false;


  ann->destroy();


  delete [] h_input;
  delete [] input;
  delete [] output;
  delete [] topology;
  delete ann;
  return true;
}


bool test_rnn_feedforward_full(){
  int M = 2
  int V = 4;
  int I = 2;
  Topology **topology = new Topology*[V];
  for(int v = 0; v < V; v++){
    topology[v] = new Topology();
    topology[v]->addLayer(I);
    topology[v]->addLayer(M);
  }

  WeightIO* weightIO1 = new WeightIO(topology[0], M);

  weightIO1->setWeight(0, 0, 0, 0.1); // warr[idx++] = 0.5;
  weightIO1->setWeight(0, 0, 1, 0.2); // warr[idx++] = 0.2;

  weightIO1->setWeight(0, 1, 0, 0.3); // warr[idx++] = 0.5;
  weightIO1->setWeight(0, 1, 1, 0.4); // warr[idx++] = 0.2;

  weightIO1->setWeight(0, 2, 0, 0.5); // warr[idx++] = 0.5;
  weightIO1->setWeight(0, 2, 1, 0.6); // warr[idx++] = 0.2;

  weightIO1->setHWeight(0, 0, 0.1); // wharr[idxh++] = 0.3;
  weightIO1->setHWeight(0, 1, 0.2); // wharr[idxh++] = 0.1;

  weightIO1->setHWeight(1, 0, 0.3); // wharr[idxh++] = 0.3;
  weightIO1->setHWeight(1, 1, 0.4); // wharr[idxh++] = 0.1;




  WeightIO* weightIO2 = new WeightIO(topology[1], M);

  weightIO2->setWeight(0, 0, 0, 0.12); // warr[idx++] = 0.5;
  weightIO2->setWeight(0, 0, 1, 0.23); // warr[idx++] = 0.2;

  weightIO2->setWeight(0, 1, 0, 0.31); // warr[idx++] = 0.5;
  weightIO2->setWeight(0, 1, 1, 0.45); // warr[idx++] = 0.2;

  weightIO2->setWeight(0, 2, 0, 0.53); // warr[idx++] = 0.5;
  weightIO2->setWeight(0, 2, 1, 0.62); // warr[idx++] = 0.2;

  weightIO2->setHWeight(0, 0, 0.51); // wharr[idxh++] = 0.3;
  weightIO2->setHWeight(0, 1, 0.62); // wharr[idxh++] = 0.1;

  weightIO2->setHWeight(1, 0, 0.83); // wharr[idxh++] = 0.3;
  weightIO2->setHWeight(1, 1, 0.24); // wharr[idxh++] = 0.1;



  WeightIO* weightIO3 = new WeightIO(topology[2], M);

  weightIO3->setWeight(0, 0, 0, 0.31); // warr[idx++] = 0.5;
  weightIO3->setWeight(0, 0, 1, 0.2); // warr[idx++] = 0.2;

  weightIO3->setWeight(0, 1, 0, 0.43); // warr[idx++] = 0.5;
  weightIO3->setWeight(0, 1, 1, 0.45); // warr[idx++] = 0.2;

  weightIO3->setWeight(0, 2, 0, 0.52); // warr[idx++] = 0.5;
  weightIO3->setWeight(0, 2, 1, 0.16); // warr[idx++] = 0.2;

  weightIO3->setHWeight(0, 0, 0.12); // wharr[idxh++] = 0.3;
  weightIO3->setHWeight(0, 1, 0.2); // wharr[idxh++] = 0.1;

  weightIO3->setHWeight(1, 0, 0.33); // wharr[idxh++] = 0.3;
  weightIO3->setHWeight(1, 1, 0.44); // wharr[idxh++] = 0.1;



  WeightIO* weightIO4 = new WeightIO(topology[3], M);

  weightIO4->setWeight(0, 0, 0, 0.12); // warr[idx++] = 0.5;
  weightIO4->setWeight(0, 0, 1, 0.2); // warr[idx++] = 0.2;

  weightIO4->setWeight(0, 1, 0, 0.34); // warr[idx++] = 0.5;
  weightIO4->setWeight(0, 1, 1, 0.4); // warr[idx++] = 0.2;

  weightIO4->setWeight(0, 2, 0, 0.52); // warr[idx++] = 0.5;
  weightIO4->setWeight(0, 2, 1, 0.6); // warr[idx++] = 0.2;

  weightIO4->setHWeight(0, 0, 0.1); // wharr[idxh++] = 0.3;
  weightIO4->setHWeight(0, 1, 0.23); // wharr[idxh++] = 0.1;

  weightIO4->setHWeight(1, 0, 0.35); // wharr[idxh++] = 0.3;
  weightIO4->setHWeight(1, 1, 0.4); // wharr[idxh++] = 0.1;

  RnnCell *rnnCell = new RnnCell(M, topology);



  rnnCell->getANN(0)->setWeights(weightIO1->getWeights(), weightIO1->getHWeights());
  rnnCell->getANN(1)->setWeights(weightIO2->getWeights(), weightIO2->getHWeights());
  rnnCell->getANN(2)->setWeights(weightIO3->getWeights(), weightIO3->getHWeights());
  rnnCell->getANN(3)->setWeights(weightIO4->getWeights(), weightIO4->getHWeights());


  Rnn* rnn = new Rnn(I,N,rnnCell)


  DerivIO* derivIO_in0 = new DerivIO(topology[0], M);
  DerivIO* derivIO_in1 = new DerivIO(topology[1], M);
  DerivIO* derivIO_in2 = new DerivIO(topology[2], M);
  DerivIO* derivIO_in3 = new DerivIO(topology[3], M);

  DerivIO* derivIO_out0 = new DerivIO(topology[0], M);
  DerivIO* derivIO_out1 = new DerivIO(topology[1], M);
  DerivIO* derivIO_out2 = new DerivIO(topology[2], M);
  DerivIO* derivIO_out3 = new DerivIO(topology[3], M);

  ///                   s  i  j  k  val
  derivIO_in0->setDeriv(0, 0, 0, 0, -0.05);
  derivIO_in0->setDeriv(0, 0, 0, 1, -0.15);

  derivIO_in0->setDeriv(0, 0, 1, 0, 0.1);
  derivIO_in0->setDeriv(0, 0, 1, 1, 0.03);

  derivIO_in0->setDeriv(0, 1, 0, 0, -0.19);
  derivIO_in0->setDeriv(0, 1, 0, 1, 0.31);

  derivIO_in0->setDeriv(0, 1, 1, 0, 0.17);
  derivIO_in0->setDeriv(0, 1, 1, 1, 0.4);

  derivIO_in0->setDeriv(0, 2, 0, 0, -0.7);
  derivIO_in0->setDeriv(0, 2, 0, 1, 0.1);

  derivIO_in0->setDeriv(0, 2, 1, 0, 0.23);
  derivIO_in0->setDeriv(0, 2, 1, 1, 0.39);


  derivIO_in0->setHDeriv(0, 0, 0, 0.71);
  derivIO_in0->setHDeriv(0, 0, 1, 0.35);

  derivIO_in0->setHDeriv(0, 1, 0, 0.21);
  derivIO_in0->setHDeriv(0, 1, 1, -0.11);

  derivIO_in0->setHDeriv(1, 0, 0, 0.2);
  derivIO_in0->setHDeriv(1, 0, 1, -0.63);

  derivIO_in0->setHDeriv(1, 1, 0, 0.1);
  derivIO_in0->setHDeriv(1, 1, 1, 0.01);



  derivIO_in1->setDeriv(0, 0, 0, 0, 0.05);
  derivIO_in1->setDeriv(0, 0, 0, 1, 0.15);

  derivIO_in1->setDeriv(0, 0, 1, 0, 0.5);
  derivIO_in1->setDeriv(0, 0, 1, 1, 0.13);

  derivIO_in1->setDeriv(0, 1, 0, 0, -0.18);
  derivIO_in1->setDeriv(0, 1, 0, 1, -0.31);

  derivIO_in1->setDeriv(0, 1, 1, 0, 0.32);
  derivIO_in1->setDeriv(0, 1, 1, 1, 0.02);

  derivIO_in1->setDeriv(0, 2, 0, 0, -0.22);
  derivIO_in1->setDeriv(0, 2, 0, 1, 0.14);

  derivIO_in1->setDeriv(0, 2, 1, 0, 0.43);
  derivIO_in1->setDeriv(0, 2, 1, 1, 0.34);


  derivIO_in1->setHDeriv(0, 0, 0, -0.71);
  derivIO_in1->setHDeriv(0, 0, 1, 0.53);

  derivIO_in1->setHDeriv(0, 1, 0, 0.35);
  derivIO_in1->setHDeriv(0, 1, 1, -0.35);

  derivIO_in1->setHDeriv(1, 0, 0, 0.2);
  derivIO_in1->setHDeriv(1, 0, 1, -0.24);

  derivIO_in1->setHDeriv(1, 1, 0, 0.22);
  derivIO_in1->setHDeriv(1, 1, 1, 0.01);







  derivIO_in2->setDeriv(0, 0, 0, 0, 0.045);
  derivIO_in2->setDeriv(0, 0, 0, 1, 0.125);

  derivIO_in2->setDeriv(0, 0, 1, 0, 0.51);
  derivIO_in2->setDeriv(0, 0, 1, 1, 0.133);

  derivIO_in2->setDeriv(0, 1, 0, 0, -0.1);
  derivIO_in2->setDeriv(0, 1, 0, 1, -0.3);

  derivIO_in2->setDeriv(0, 1, 1, 0, 0.32);
  derivIO_in2->setDeriv(0, 1, 1, 1, 0.2);

  derivIO_in2->setDeriv(0, 2, 0, 0, -0.2);
  derivIO_in2->setDeriv(0, 2, 0, 1, 0.4);

  derivIO_in2->setDeriv(0, 2, 1, 0, 0.443);
  derivIO_in2->setDeriv(0, 2, 1, 1, 0.374);


  derivIO_in2->setHDeriv(0, 0, 0, -0.751);
  derivIO_in2->setHDeriv(0, 0, 1, 0.523);

  derivIO_in2->setHDeriv(0, 1, 0, 0.35);
  derivIO_in2->setHDeriv(0, 1, 1, -0.35);

  derivIO_in2->setHDeriv(1, 0, 0, 0.23);
  derivIO_in2->setHDeriv(1, 0, 1, -0.241);

  derivIO_in2->setHDeriv(1, 1, 0, 0.226);
  derivIO_in2->setHDeriv(1, 1, 1, 0.015);








  derivIO_in3->setDeriv(0, 0, 0, 0, 0.105);
  derivIO_in3->setDeriv(0, 0, 0, 1, 0.135);

  derivIO_in3->setDeriv(0, 0, 1, 0, 0.51);
  derivIO_in3->setDeriv(0, 0, 1, 1, 0.613);

  derivIO_in3->setDeriv(0, 1, 0, 0, -0.188);
  derivIO_in3->setDeriv(0, 1, 0, 1, -0.321);

  derivIO_in3->setDeriv(0, 1, 1, 0, 0.362);
  derivIO_in3->setDeriv(0, 1, 1, 1, 0.072);

  derivIO_in3->setDeriv(0, 2, 0, 0, -0.122);
  derivIO_in3->setDeriv(0, 2, 0, 1, 0.124);

  derivIO_in3->setDeriv(0, 2, 1, 0, 0.463);
  derivIO_in3->setDeriv(0, 2, 1, 1, 0.354);


  derivIO_in3->setHDeriv(0, 0, 0, -0.371);
  derivIO_in3->setHDeriv(0, 0, 1, 0.523);

  derivIO_in3->setHDeriv(0, 1, 0, 0.325);
  derivIO_in3->setHDeriv(0, 1, 1, -0.535);

  derivIO_in3->setHDeriv(1, 0, 0, 0.12);
  derivIO_in3->setHDeriv(1, 0, 1, -0.224);

  derivIO_in3->setHDeriv(1, 1, 0, 0.226);
  derivIO_in3->setHDeriv(1, 1, 1, 0.051);



  Derivatives **deriv_in = new Derivatives*[V];
  Derivatives **deriv_out = new Derivatives*[V];

  deriv_in[0] =  derivIO_in0->getDerivatives();
  deriv_in[1] =  derivIO_in1->getDerivatives();
  deriv_in[2] =  derivIO_in2->getDerivatives();
  deriv_in[3] =  derivIO_in3->getDerivatives();

  deriv_out[0] =  derivIO_out0->getDerivatives();
  deriv_out[1] =  derivIO_out1->getDerivatives();
  deriv_out[2] =  derivIO_out2->getDerivatives();
  deriv_out[3] =  derivIO_out3->getDerivatives();








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
  // delete [] warr1;
  // delete [] wharr1;
  //
  // delete [] warr2;
  // delete [] wharr2;
  //
  // delete [] warr3;
  // delete [] wharr3;
  //
  // delete [] warr4;
  // delete [] wharr4;

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

// bool test__char_to_vec(){
//
//     double *vec0 =  char_to_vec(" abcd", ' ');
//     double *vec1 =  char_to_vec(" abcd", 'b');
//     double *vec2 =  char_to_vec(" abcd", 'd');
//
//     printf("vec0 : ");
//     for(int i = 0; i < 5; i++)
//       printf("%.1f; ", vec0[i]);
//     printf("\n");
//
//     printf("vec1 : ");
//     for(int i = 0; i < 5; i++)
//       printf("%.1f; ", vec1[i]);
//     printf("\n");
//
//     printf("vec2 : ");
//     for(int i = 0; i < 5; i++)
//       printf("%.1f; ", vec2[i]);
//     printf("\n");
//
//     return true;
// }

bool test__vec_to_char(){

    double *vec = new double[5];
    vec[0] = 0.1;
    vec[1] = 0.1;
    vec[2] = 0.5;
    vec[3] = 0.1;
    vec[4] = 0.2;

    char c = vec_to_char(" abcd", vec);
    printf("c = %c\n", c);

return true;
}

bool test__str_to_nodes(){
  char abc[64]=" abcd";

  DataNode* node = str_to_nodes(abc, "cc dba");
  DataNode* q = node;
  int index = 0;
  do{
    printf("[%d] : %c\n", index++, vec_to_char(abc, q->vec));
    q = q->next;
  }while(q != NULL);

  char str[64]="";
  nodes_to_str(abc, node, str);

  printf("got back: |%s|\n", str);

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

  // passed = test_rnn_feedforward(); failCount += passed ? 0 : 1;
  // printf("%s - test_rnn_feedforwards_of_networks\n", passed ? "PASSED" : "FAILED");
  // printf("%s\n", "---------------------");
  // passed = test_rnn_feedforward_full(); failCount += passed ? 0 : 1;
  // printf("%s - test_rnn_feedforwards_full\n", passed ? "PASSED" : "FAILED");


  passed = test_backprogg(); failCount += passed ? 0 : 1;
  printf("%s - test_backprogg\n", passed ? "PASSED" : "FAILED");

  // passed = test__char_to_vec(); failCount += passed ? 0 : 1;
  // printf("%s - test__char_to_vec\n", passed ? "PASSED" : "FAILED");

  passed = test__vec_to_char(); failCount += passed ? 0 : 1;
  printf("%s - test__vec_to_char\n", passed ? "PASSED" : "FAILED");


  passed = test__str_to_nodes(); failCount += passed ? 0 : 1;
  printf("%s - test__str_to_nodes\n", passed ? "PASSED" : "FAILED");




  printf("\n");
  if(failCount == 0) printf("ALL tests PASSED\n");
  else printf("%d TESTS FAILED\n", failCount);

  return failCount == 0;
}
