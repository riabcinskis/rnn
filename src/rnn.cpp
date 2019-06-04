#include "rnn.h"

/* C++ code here. */
void RnnConfig::setTopologies(Topology **top){
  cTopology = top;
}

void RnnConfig::setM(int M){
  mM=M;
}


Topology** RnnConfig::getTopologies(){
  return cTopology;
}

int RnnConfig::getM(){
  return mM;
}


//
// RnnCell
//

RnnCell::RnnCell(int M, string filename) {
  FILE * p1File;
  p1File = fopen(filename.c_str(), "rb");

  Topology** top= new Topology*[4];
  top[0] = new Topology();
  top[1] = new Topology();
  top[2] = new Topology();
  top[3] = new Topology();
  top[0]->readTopology(p1File);
  top[1]->readTopology(p1File);
  top[2]->readTopology(p1File);
  top[3]->readTopology(p1File);


  prepare(M,top);

  double* Ww0= new double[top[0]->obtainWeightCount()];
  double* Ww1= new double[top[1]->obtainWeightCount()];
  double* Ww2= new double[top[2]->obtainWeightCount()];
  double* Ww3= new double[top[3]->obtainWeightCount()];

  double* Whw0= new double[M*top[0]->getLayerSize(1)];
  double* Whw1= new double[M*top[1]->getLayerSize(1)];
  double* Whw2= new double[M*top[2]->getLayerSize(1)];
  double* Whw3= new double[M*top[3]->getLayerSize(1)];

  // (void)fread (Ww0 , sizeof(double), top[0]->obtainWeightCount(), p1File);
  // (void)fread (Whw0 , sizeof(double), M*top[0]->getLayerSize(1), p1File);
  //
  // (void)fread (Ww1 , sizeof(double), top[1]->obtainWeightCount(), p1File);
  // (void)fread (Whw1 , sizeof(double), M*top[1]->getLayerSize(1), p1File);
  //
  // (void)fread (Ww2 , sizeof(double), top[2]->obtainWeightCount(), p1File);
  // (void)fread (Whw2 , sizeof(double), M*top[2]->getLayerSize(1), p1File);
  //
  // (void)fread (Ww3 , sizeof(double), top[3]->obtainWeightCount(), p1File);
  // (void)fread (Whw3 , sizeof(double), M*top[3]->getLayerSize(1), p1File);

  size_t result = fread (Ww0 , sizeof(double), top[0]->obtainWeightCount(), p1File);
  result = fread (Whw0 , sizeof(double), M*top[0]->getLayerSize(1), p1File);

  result = fread (Ww1 , sizeof(double), top[1]->obtainWeightCount(), p1File);
  result = fread (Whw1 , sizeof(double), M*top[1]->getLayerSize(1), p1File);

  result = fread (Ww2 , sizeof(double), top[2]->obtainWeightCount(), p1File);
  result = fread (Whw2 , sizeof(double), M*top[2]->getLayerSize(1), p1File);

  result = fread (Ww3 , sizeof(double), top[3]->obtainWeightCount(), p1File);
  result = fread (Whw3 , sizeof(double), M*top[3]->getLayerSize(1), p1File);


  anns[0]->setWeights(Ww0,Whw0);

  anns[1]->setWeights(Ww1,Whw1);

  anns[2]->setWeights(Ww2,Whw2);

  anns[3]->setWeights(Ww3,Whw3);



  init(p1File);
  fclose (p1File);
};


void RnnCell::printf_Network(string output_filename){

    FILE * pFile;
    const char * c = output_filename.c_str();
    pFile = fopen(c, "wb");
    anns[0]->getTopology()->printTopology(pFile);
    anns[1]->getTopology()->printTopology(pFile);
    anns[2]->getTopology()->printTopology(pFile);
    anns[3]->getTopology()->printTopology(pFile);

    fwrite (anns[0]->getWeights() , sizeof(double), anns[0]->getTopology()->obtainWeightCount(), pFile);
    fwrite (anns[0]->getHWeights() , sizeof(double), anns[0]->getTopology()->getLayerSize(1)*M, pFile);

    fwrite (anns[1]->getWeights() , sizeof(double), anns[1]->getTopology()->obtainWeightCount(), pFile);
    fwrite (anns[1]->getHWeights() , sizeof(double), anns[1]->getTopology()->getLayerSize(1)*M, pFile);

    fwrite (anns[2]->getWeights() , sizeof(double), anns[2]->getTopology()->obtainWeightCount(), pFile);
    fwrite (anns[2]->getHWeights() , sizeof(double), anns[2]->getTopology()->getLayerSize(1)*M, pFile);

    fwrite (anns[3]->getWeights() , sizeof(double), anns[3]->getTopology()->obtainWeightCount(), pFile);
    fwrite (anns[3]->getHWeights() , sizeof(double), anns[3]->getTopology()->getLayerSize(1)*M, pFile);

    fclose (pFile);
}

void RnnCell::prepare(int M, Topology **top){
  //printf("  RnnCell::prepare : M = %d\n", M);

  this->M = M;
  V = 4;
  //printf("(2)RnnCell::prepare : M = %d\n", M);

  double (*func)(double);
  double (*func_deriv)(double);

  double (*func_tanh)(double);
  double (*func_tanh_deriv)(double);

  func=f;
  func_deriv = f_deriv;

  func_tanh=f_tanh;
  func_tanh_deriv = f_tanh_deriv;


  anns = new AnnSerial*[V];
  for(int v = 0; v < V; v++){
    if(v==2){
      anns[v] = new AnnSerial(V, v, M, top, func_tanh, func_tanh_deriv);
    } else {
      anns[v] = new AnnSerial(V, v, M, top, func, func_deriv);
    }
  }

  ann_forget = anns[0];
  ann_input = anns[1];
  ann_gate = anns[2];
  ann_output = anns[3];




  b = new double[M];
  c_current = new double[M];
  c_new = new double[M];
  h_current = new double[M];
  h_new = new double[M];

  a_outputs = new double*[V];
  for(int v = 0; v < V; v++){
    a_outputs[v] = new double[M];
  }

  aderiv = new Derivatives**[V];
  for(int u = 0; u < V; u++){
    aderiv[u] = new Derivatives*[V];
    for(int v = 0; v < V; v++){
      aderiv[u][v] = new Derivatives;
      aderiv[u][v]->v = new double[top[v]->obtainWeightCount()*M];
      aderiv[u][v]->vh = new double[M*top[v]->getLayerSize(1)*M];
    }
  }
}


void RnnCell::init(FILE * pFile=NULL){

//  printf("  RnnCell::init : M = %d\n", M);

  for(int i=0; i<M; i++){
    c_current[i] = 0;
    h_current[i] = 0;
  }
}

void RnnCell::feedForward(double *h_in, double *c_in, double *a_in, double *c_out, double *h_out){

  c_current = c_in; // c_t0, c_t1,
  h_current = h_in;

  for(int v = 0; v < V; v++)
    anns[v]->feedForward(h_in, a_in, a_outputs[v]);

  for(int i=0; i < M; i++){
    c_new[i] = c_out[i] = c_in[i] * a_outputs[0][i] + a_outputs[1][i] * a_outputs[2][i];
    b[i] = tanh(c_out[i])* a_outputs[3][i];
  }

  double sumB = 0;
  for(int i = 0; i < M; i++)
    sumB += exp(b[i]);

  for(int i = 0; i < M; i++)
    h_new[i] = h_out[i] = exp(b[i]) / sumB;

}


void RnnCell::backPropagation(RnnDerivatives *deriv_in, RnnDerivatives *deriv_out){



  for(int u = 0; u < V; u++)
    anns[u]->backPropagation(deriv_in->hderiv, aderiv[u]);



  for(int v = 0; v < V; v++){

    Topology *vtop = anns[v]->getTopology();
    for(int s = 0; s < vtop->getLayerCount()-1; s++){

      for(int wi = 0; wi < vtop->getLayerSize(s)+1; wi++){

        for(int wj = 0; wj < vtop->getLayerSize(s+1); wj++){


          for(int k = 0; k < M; k++){

            double cderiv = deriv_in->cderiv[v]->v[anns[v]->vi(v, s, wi, wj, k)];
            // printf("cderiv: %.20f\n", cderiv);
            double a_forget = ann_forget->getOutput(k);
            // printf("a_forget: %.20f\n", a_forget);
            double c = c_current[k];
            // printf("c: %.20f\n", c);
            double a_forget_deriv = aderiv[0][v]->v[anns[v]->vi(v, s, wi, wj, k)];
            // printf("a_forget_deriv: %.20f\n", a_forget_deriv);
            double a_input_deriv = aderiv[1][v]->v[anns[v]->vi(v, s, wi, wj, k)];
            // printf("a_input_deriv: %.20f\n", a_input_deriv);
            double a_gate = ann_gate->getOutput(k);
            // printf("a_gate: %.20f\n", a_gate);
            double a_input = ann_input->getOutput(k);
            // printf("a_input: %.20f\n", a_input);
            double a_gate_deriv = aderiv[2][v]->v[anns[v]->vi(v, s, wi, wj, k)];
            // printf("a_gate_deriv: %.20f\n", a_gate_deriv);

            deriv_out->cderiv[v]->v[anns[v]->vi(v, s, wi, wj, k)] = cderiv*a_forget + c*a_forget_deriv + a_input_deriv*a_gate + a_input*a_gate_deriv;
            // printf("cderiv*a_forget:  %.20f\n", cderiv*a_forget);
            // printf("c*a_forget_deriv:  %.20f\n", c*a_forget_deriv);
            // printf("a_input_deriv*a_gate:  %.20f\n", a_input_deriv*a_gate);
            // printf("a_input*a_gate_deriv:  %.20f\n", a_input*a_gate_deriv);
            // printf("cia:  %.20f\n", deriv_out->cderiv[0]->v[0]);
            // printf("cia:  %.20f\n", deriv_out->cderiv[0]->v[1]);
            // break;
          }
        }
      }
    }
    // printf("cia:  %.20f\n", deriv_out->cderiv[0]->v[0]);
    // printf("cia:  %.20f\n", deriv_out->cderiv[0]->v[1]);


    for(int wi = 0; wi < M /*vtop->getLayerSize(0)*/; wi++){
      for(int wj = 0; wj < vtop->getLayerSize(1); wj++){

        for(int k = 0; k < M; k++){

          double cderiv = deriv_in->cderiv[v]->vh[anns[v]->vhi(v, wi, wj, k)];
          // if(wi==1 && wj==0&& k==0 && v==0) printf("cderiv: %.20f\n", cderiv);

          double a_forget = ann_forget->getOutput(k);
          // if(wi==1 && wj==0&& k==0&& v==0) printf("a_forget: %.20f\n", a_forget);

          double c = c_current[k];
          // if(wi==1 && wj==0&& k==0&& v==0) printf("c: %.20f\n", c);

          double a_forget_deriv = aderiv[0][v]->vh[anns[v]->vhi(v, wi, wj, k)];
          // if(wi==1 && wj==0&& k==0&& v==0) printf("a_forget_deriv: %.20f\n", a_forget_deriv);

          double a_input_deriv = aderiv[1][v]->vh[anns[v]->vhi(v, wi, wj, k)];
          // if(wi==1 && wj==0&& k==0&& v==0) printf("a_input_deriv: %.20f\n", a_input_deriv);

          double a_gate = ann_gate->getOutput(k);
          // if(wi==1 && wj==0&& k==0&& v==0) printf("a_gate: %.20f\n", a_gate);

          double a_input = ann_input->getOutput(k);
          // if(wi==1 && wj==0&& k==0&& v==0) printf("a_input: %.20f\n", a_input);

          double a_gate_deriv = aderiv[2][v]->vh[anns[v]->vhi(v, wi, wj, k)];
          // if(wi==1 && wj==0&& k==0&& v==0) printf("a_gate_deriv: %.20f\n", a_gate_deriv);

          deriv_out->cderiv[v]->vh[anns[v]->vhi(v, wi, wj, k)] = cderiv*a_forget + c*a_forget_deriv + a_input_deriv*a_gate + a_input*a_gate_deriv;
          // if(wi==1 && wj==0 && k==0&& v==0) printf("cderiv*a_forget:  %.20f\n", cderiv*a_forget);
          // if(wi==1 && wj==0 && k==0&& v==0) printf("c*a_forget_deriv:  %.20f\n", c*a_forget_deriv);
          // if(wi==1 && wj==0 && k==0&& v==0) printf("a_input_deriv*a_gate:  %.20f\n", a_input_deriv*a_gate);
          // if(wi==1 && wj==0 && k==0&& v==0) printf("a_input*a_gate_deriv:  %.20f\n", a_input*a_gate_deriv);
          // if(wi==1 && wj==0 && k==0&& v==0) printf("deriv_out->cderiv[v]->vh[anns[v]->vhi(v, wi, wj, k)]: %.20f\n", deriv_out->cderiv[v]->vh[anns[v]->vhi(v, wi, wj, k)]);
        }
      }
    }
  }

  // printf("1%.20f\n", deriv_out->cderiv[0]->vh[0]);
  // printf("%.20f\n", deriv_out->cderiv[0]->vh[1]);
  // printf("%.20f\n", deriv_out->cderiv[0]->vh[2]);
  // printf("4%.20f\n", deriv_out->cderiv[0]->vh[3]);
  double* sm_deriv = new double[M*M];


  for(int k = 0; k < M; k++)
    for(int n = 0; n < M; n++)
      if(k == n) sm_deriv[k*M + n] = h_new[k]*(1-h_new[k]);
      else sm_deriv[k*M + n] = -h_new[k]*h_new[n];

      // printf("sm_deriv[k*M]: %.20f\n", sm_deriv[0*M]);
      // printf("sm_deriv[k*M]: %.20f\n", sm_deriv[1]);

      for(int v = 0; v < V; v++){
        Topology *vtop = anns[v]->getTopology();
        for(int s = 0; s < vtop->getLayerCount()-1; s++){
          for(int wi = 0; wi < vtop->getLayerSize(s)+1; wi++){
            for(int wj = 0; wj < vtop->getLayerSize(s+1); wj++){
              for(int k = 0; k < M; k++){
                double sum = 0;
                for(int n = 0; n < M; n++){
                  double sum2=0;
                  //sm_deriv[k*M + n]
                  double a_output_deriv = aderiv[3][v]->v[anns[v]->vi(v, s, wi, wj, n)];
                  // printf("a_output_deriv: %.20f\n", a_output_deriv);
                  double a_output = ann_output->getOutput(n);
                  // printf("a_output: %.20f\n", a_output);
                  double c_deriv = deriv_out->cderiv[v]->v[anns[v]->vi(v, s, wi, wj, n)];
                  // printf("c_deriv: %.20f\n", c_deriv);
                  sum2 += a_output_deriv*f_tanh(c_new[n]) + a_output*f_tanh_deriv(c_new[n])*c_deriv;
                  // printf("a_output_deriv*f_tanh(c_new[n]): %.20f\n", a_output_deriv*f_tanh(c_new[n]));
                  // printf("a_output*f_tanh_deriv(c_new[n])*c_deriv: %.20f\n", a_output*f_tanh_deriv(c_new[n])*c_deriv);
                  sum2 *= sm_deriv[k*M+n];

                  sum+=sum2;

                  // printf("sum: %.20f\n", sum);
                }
                deriv_out->hderiv[v]->v[anns[v]->vi(v, s, wi, wj, k)] = sum;

              }
            }
          }
        }


        for(int wi = 0; wi < M /*vtop->getLayerSize(0)*/; wi++){
          for(int wj = 0; wj < vtop->getLayerSize(1); wj++){
            for(int k = 0; k < M; k++){
              double sum = 0;
              for(int n = 0; n < M; n++){
                //sm_deriv[k*M + n]
                double a_output_deriv = aderiv[3][v]->vh[anns[v]->vhi(v, wi, wj, n)];
                // printf("a_output_deriv:%.20f\n", a_output_deriv);

                double a_output = ann_output->getOutput(n);
                // printf("a_output:%.20f\n", a_output);
                double c_deriv = deriv_out->cderiv[v]->vh[anns[v]->vhi(v, wi, wj, n)];
                // printf("c_deriv:%.20f\n", c_deriv);
                sum += a_output_deriv*f_tanh(c_new[n]) + a_output*f_tanh_deriv(c_new[n])*c_deriv;
                sum *= sm_deriv[k*M+n];
              }
              deriv_out->hderiv[v]->vh[anns[v]->vhi(v, wi, wj, k)] = sum;

            }
          }
        }
      }

  delete [] sm_deriv;

}

void RnnCell::destroy(){
    for(int i = 0; i < V; i++){
      anns[i]->destroy();
    }

   delete [] anns;
   anns = NULL;


  delete c_current;
  c_current = NULL;
  delete c_new;
  c_new = NULL;
  delete h_current;
  h_current = NULL;
  delete h_new;
  h_new = NULL;
  delete b;
  b = NULL;
  delete [] a_outputs;
  a_outputs = NULL;

  delete ann_forget; // 0
  delete ann_input; // 1
  delete ann_gate; // 2
  delete ann_output; // 3

  delete *aderiv;

}


AnnSerial* RnnCell::getANN(int v){
  return anns[v];
}


//
// SecondMarkLimit
//

SecondMarkLimit::SecondMarkLimit(int markIndex, int M){
  this->markIndex = markIndex;
  this->M = M;
}

void SecondMarkLimit::reset(){
  count = 0;
}

bool SecondMarkLimit::check(double *vec){
  int maxAt = 0;
  for(int i = 1; i < M; i++)
    maxAt = vec[maxAt] < vec[i] ? i : maxAt;

  if(maxAt == markIndex) count++;

  if(count == 2) return true;
  return false;
}


//
// DataNode
//
DataNode::DataNode(int M){
  vec = new double[M];
  next = NULL;
}

//
// Rnn
//

Rnn::Rnn(int I, int M, RnnCell *rnnCell){

  impl = RNN_FULL_BACKPROPAGATION;
  // printf("M:%d\n", M);
  Rnn(I, M, rnnCell, impl);
}

Rnn::Rnn(int I, int M, RnnCell *rnnCell, int impl){

  this->I = I;
  this->M = M;
  this->V = 4;
  cRnnCell = rnnCell;
  this->impl = impl;
  // printf("M:%d\n", this->M);
  h_in = new double[M];
  h_out = new double[M];
  c_in = new double[M];
  c_out = new double[M];
  // printf("%s\n", "buvo");
  rnnDeriv = new RnnDerivatives*[2];
  rnnDeriv[0] = new RnnDerivatives;
  rnnDeriv[1] = new RnnDerivatives;
  allocateRnnDerivatives(rnnDeriv[0]);
  allocateRnnDerivatives(rnnDeriv[1]);

  errDeriv = new ErrorDerivatives*[V];

  for(int v = 0; v < V; v++){
    errDeriv[v] = new ErrorDerivatives;
    errDeriv[v]->v = new double[cRnnCell->getANN(v)->getTopology()->obtainWeightCount()];
    errDeriv[v]->vh = new double[cRnnCell->getANN(v)->getTopology()->getLayerSize(1)*M];
  }
}

DataNode* Rnn::feedForward(DataNode* input, OutputLimit *outputLimit){
  // printf("%d\n", M);
  for(int k = 0; k < M; k++){
    h_in[k] = 0;
    c_in[k] = 0;
  }

  DataNode* p = input;
  while(p->next != NULL){


    cRnnCell->feedForward(h_in, c_in, p->vec, c_out, h_out);
    copyVector(h_in, h_out, M);
    copyVector(c_in, c_out, M);

    p = p->next;
  }

  DataNode* out = new DataNode(M);
  DataNode* q = out;

  outputLimit->reset();

  cRnnCell->feedForward(h_in, c_in, p->vec, c_out, h_out);
  copyVector(h_in, h_out, M);
  copyVector(c_in, c_out, M);
  copyVector(q->vec, h_out, M);

  if(outputLimit->check(q->vec)) return out;

 double* empty_input = new double[I];
 for(int i = 0; i < I; i++)
  empty_input[i] = 0;


  int count=0;
  do{
    count++;
    // printf("%d\n", count);
    cRnnCell->feedForward(h_in, c_in, empty_input, c_out, h_out);
    copyVector(h_in, h_out, M);
    copyVector(c_in, c_out, M);
    // printf("%s\n", "asdsa");
    q->next = new DataNode(M);
    q = q->next;
    copyVector(q->vec, h_out, M);




  }while(outputLimit->check(q->vec) == false && count < 100);

  delete [] empty_input;

  return out;
}

bool Rnn::backPropagation(DataNode* input, DataNode* output, OutputLimit *outputLimit, double &error){

  for(int k = 0; k < M; k++){
    h_in[k] = 0;
    c_in[k] = 0;
  }

  error = 0;

  outputLimit->reset();

  int derivIndex = 0;
  resetHDerivatives(rnnDeriv[derivIndex]->hderiv);
  resetCDerivatives(rnnDeriv[derivIndex]->cderiv);

  //initRnnDerivatives(rnnDeriv[1 - derivIndex]);

  DataNode* p = input;


//printf("----------------\n");

  while(p->next != output){
    //printf("A");
    cRnnCell->feedForward(h_in, c_in, p->vec, c_out, h_out);
    if(impl == RNN_APPROX_BACKPROPAGATION)resetHDerivatives(rnnDeriv[derivIndex]->hderiv);
    cRnnCell->backPropagation(rnnDeriv[derivIndex], rnnDeriv[1-derivIndex]);

    copyVector(h_in, h_out, M);
    copyVector(c_in, c_out, M);
    p = p->next;
    derivIndex = 1 - derivIndex;
  }



  cRnnCell->feedForward(h_in, c_in, p->vec, c_out, h_out);
  if(impl == RNN_APPROX_BACKPROPAGATION)resetHDerivatives(rnnDeriv[derivIndex]->hderiv);
  cRnnCell->backPropagation(rnnDeriv[derivIndex], rnnDeriv[1-derivIndex]);
  copyVector(h_in, h_out, M);
  copyVector(c_in, c_out, M);

  sumErrorDerivatives(h_out, rnnDeriv[1-derivIndex]->hderiv, output->vec);
  error += calcError(h_out, output->vec);
  int outputCount = 1;


  derivIndex = 1 - derivIndex;
  DataNode* q = output;

  double* empty_input = new double[I];
  for(int i = 0; i < I; i++)
   empty_input[i] = 0;

   //char abc[64]=" abcdefghijklmnopqrstuvwxyz-";
   //char abc[64]=" 0123456789";

   do{

     q = q->next;
     //printf("%c", vec_to_char(abc, q->vec));
    // printf("B");
     if(q == NULL) {

       delete [] empty_input;
       return false;
     }

     cRnnCell->feedForward(h_in, c_in, empty_input, c_out, h_out);
     if(impl == RNN_APPROX_BACKPROPAGATION)resetHDerivatives(rnnDeriv[derivIndex]->hderiv);
     cRnnCell->backPropagation(rnnDeriv[derivIndex], rnnDeriv[1-derivIndex]);
     copyVector(h_in, h_out, M);
     copyVector(c_in, c_out, M);

     sumErrorDerivatives(h_out, rnnDeriv[1-derivIndex]->hderiv, q->vec);
     error += calcError(h_out, q->vec);
     outputCount++;

     derivIndex = 1 - derivIndex;

   }while(outputLimit->check(q->vec) == false);


   delete [] empty_input;

   error = error / (double)outputCount;

   return true;

}

void Rnn::updateWeights(double alpha, double eta){
  for(int v = 0; v < V; v++)
    cRnnCell->getANN(v)->updateWeights(errDeriv[v], alpha, eta);
}

void Rnn::resetErrorDerivatives(){
  for(int v = 0; v < V; v++){
    for(int k = 0; k < cRnnCell->getANN(v)->getTopology()->obtainWeightCount(); k++)
      errDeriv[v]->v[k] = 0.0;
    for(int k = 0; k < cRnnCell->getANN(v)->getTopology()->getLayerSize(1)*M; k++)
      errDeriv[v]->vh[k] = 0.0;
  }
}


// PRIVATE


RnnDerivatives* Rnn::allocateRnnDerivatives(RnnDerivatives* deriv){
  deriv->hderiv = new Derivatives*[V];
  deriv->cderiv = new Derivatives*[V];

  for(int v = 0; v < V; v++){
    deriv->hderiv[v] = new Derivatives;
    deriv->hderiv[v]->v = new double[cRnnCell->getANN(v)->getTopology()->obtainWeightCount()*M];
    deriv->hderiv[v]->vh = new double[cRnnCell->getANN(v)->getTopology()->getLayerSize(1)*M*M];

    deriv->cderiv[v] = new Derivatives;
    deriv->cderiv[v]->v = new double[cRnnCell->getANN(v)->getTopology()->obtainWeightCount()*M];
    deriv->cderiv[v]->vh = new double[cRnnCell->getANN(v)->getTopology()->getLayerSize(1)*M*M];
  }
}


void Rnn::resetHDerivatives(Derivatives** hderiv){
  for(int v = 0; v < V; v++){
    for(int k = 0; k < cRnnCell->getANN(v)->getTopology()->obtainWeightCount()*M; k++)
      hderiv[v]->v[k] = 0.0;

    for(int k = 0; k < cRnnCell->getANN(v)->getTopology()->getLayerSize(1)*M*M; k++)
      hderiv[v]->vh[k] = 0.0;

  }
}
void Rnn::resetCDerivatives(Derivatives** cderiv){
  for(int v = 0; v < V; v++){
    for(int k = 0; k < cRnnCell->getANN(v)->getTopology()->obtainWeightCount()*M; k++)
      cderiv[v]->v[k] = 0.0;

    for(int k = 0; k < cRnnCell->getANN(v)->getTopology()->getLayerSize(1)*M*M; k++)
      cderiv[v]->vh[k] = 0.0;

  }
}


void Rnn::copyVector(double* vec_b, double *vec_a, int n){
  for(int k = 0; k < n; k++)
    vec_b[k] = vec_a[k];
}

void Rnn::sumErrorDerivatives(double *h, Derivatives **hderiv, double *y){
  for(int v = 0; v < V; v++){
    Topology *top = cRnnCell->getANN(v)->getTopology();
    for(int s = 0; s < top->getLayerCount()-1; s++){
      for(int wi = 0; wi < top->getLayerSize(s)+1; wi++){
        for(int wj = 0; wj < top->getLayerSize(s+1); wj++){
          double sum = 0;
          for(int n = 0; n < M; n++)
            sum += (y[n]-h[n])*hderiv[v]->v[cRnnCell->getANN(v)->vi(v, s, wi, wj, n)];
          errDeriv[v]->v[cRnnCell->getANN(v)->vi(v, s, wi, wj)] = sum;
        }
      }
    }


    for(int wi = 0; wi < M; wi++){
      for(int wj = 0; wj < top->getLayerSize(1); wj++){
        double sum = 0;
        for(int n = 0; n < M; n++)
          sum += (y[n]-h[n])*hderiv[v]->vh[cRnnCell->getANN(v)->vhi(v, wi, wj, n)];
        errDeriv[v]->vh[cRnnCell->getANN(v)->vhi(v, wi, wj)] = sum;
      }
    }
  }
}

double Rnn::calcError(double *h, double *y){
  double error = 0;
  for(int k = 0; k < M; k++)
    error += 0.5*(h[k] - y[k])*(h[k] - y[k]);
  return error;
}

double f(double x){
  double y = 1 + exp(-x);
  return 1 / y;
}

double f_deriv(double x){
  return exp(-x) / pow((1 + exp(-x)), 2);
}

double f_tanh(double x){
  return tanh(x);
}

double f_tanh_deriv(double x){
  return 1 - pow(tanh(x), 2);
}
