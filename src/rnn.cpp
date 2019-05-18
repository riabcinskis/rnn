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
void RnnCell::prepare(int M, Topology **top){
  this->M = M;
  V = 4;

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
            double a_forget = ann_forget->getOutput(k);
            double c = c_current[k];
            double a_forget_deriv = aderiv[0][v]->v[anns[v]->vi(v, s, wi, wj, k)];
            double a_input_deriv = aderiv[1][v]->v[anns[v]->vi(v, s, wi, wj, k)];
            double a_gate = ann_gate->getOutput(k);
            double a_input = ann_input->getOutput(k);
            double a_gate_deriv = aderiv[2][v]->v[anns[v]->vi(v, s, wi, wj, k)];

            deriv_out->cderiv[v]->v[anns[v]->vi(v, s, wi, wj, k)] = cderiv*a_forget + c*a_forget_deriv + a_input_deriv*a_gate + a_input*a_gate_deriv;

          }
        }
      }
    }


    for(int wi = 0; wi < M /*vtop->getLayerSize(0)*/; wi++){
      for(int wj = 0; wj < vtop->getLayerSize(1); wj++){
        for(int k = 0; k < M; k++){
          double cderiv = deriv_in->cderiv[v]->vh[anns[v]->vhi(v, wi, wj, k)];
          double a_forget = ann_forget->getOutput(k);
          double c = c_current[k];
          double a_forget_deriv = aderiv[0][v]->vh[anns[v]->vhi(v, wi, wj, k)];
          double a_input_deriv = aderiv[1][v]->vh[anns[v]->vhi(v, wi, wj, k)];
          double a_gate = ann_gate->getOutput(k);
          double a_input = ann_input->getOutput(k);
          double a_gate_deriv = aderiv[2][v]->vh[anns[v]->vhi(v, wi, wj, k)];

          deriv_out->cderiv[v]->vh[anns[v]->vhi(v, wi, wj, k)] = cderiv*a_forget + c*a_forget_deriv + a_input_deriv*a_gate + a_input*a_gate_deriv;

        }
      }
    }
  }

  double* sm_deriv = new double[M*M];

  for(int k = 0; k < M; k++)
    for(int n = 0; n < M; n++)
      if(k == n) sm_deriv[k*M + n] = h_new[k]*(1-h_new[k]);
      else sm_deriv[k*M + n] = -h_new[k]*h_new[n];


      for(int v = 0; v < V; v++){
        Topology *vtop = anns[v]->getTopology();
        for(int s = 0; s < vtop->getLayerCount()-1; s++){
          for(int wi = 0; wi < vtop->getLayerSize(s)+1; wi++){
            for(int wj = 0; wj < vtop->getLayerSize(s+1); wj++){
              for(int k = 0; k < M; k++){
                double sum = 0;
                for(int n = 0; n < M; n++){
                  //sm_deriv[k*M + n]
                  double a_output_deriv = aderiv[3][v]->v[anns[v]->vi(v, s, wi, wj, n)];

                  double a_output = ann_output->getOutput(n);

                  double c_deriv = deriv_out->cderiv[v]->v[anns[v]->vi(v, s, wi, wj, n)];
                  sum += a_output_deriv*f_tanh(c_new[n]) + a_output*f_tanh_deriv(c_new[n])*c_deriv;
                  sum *= sm_deriv[k*M+n];
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

                double a_output = ann_output->getOutput(n);

                double c_deriv = deriv_out->cderiv[v]->vh[anns[v]->vhi(v, wi, wj, n)];
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
  this->I = I;
  this->M = M;
  cRnnCell = rnnCell;
  allocateRnnDerivatives(deriv_in);
  allocateRnnDerivatives(deriv_out);

  h_in = new double[M];
  h_out = new double[M];
  c_in = new double[M];
  c_out = new double[M];

}

DataNode* Rnn::feedForward(DataNode* input, OutputLimit *outputLimit){



  for(int k = 0; k < M; k++){
    h_in[k] = 0;
    c_in[k] = 0;
  }




  DataNode* p = input;
  int c = 0;
  while(p->next != NULL){
    printf("c=%d\n", c++);

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

int index = 0;
  do{
printf("index=%d\n", index++);
    cRnnCell->feedForward(h_in, c_in, empty_input, c_out, h_out);
    copyVector(h_in, h_out, M);
    copyVector(c_in, c_out, M);

    q->next = new DataNode(M);
    q = q->next;
    copyVector(q->vec, h_out, M);



if(index == 10) break;
  }while(outputLimit->check(q->vec) == false);

  delete [] empty_input;

  return out;
}

// PRIVATE
RnnDerivatives* Rnn::allocateRnnDerivatives(RnnDerivatives* deriv){

}

RnnDerivatives* Rnn::deallocateRnnDerivatives(RnnDerivatives* deriv){

}

void Rnn::initRnnDerivatives(RnnDerivatives* deriv){

}

void Rnn::copyRnnDerivatives(RnnDerivatives* deri_b, RnnDerivatives* deriv_a){

}

void Rnn::copyVector(double* vec_b, double *vec_a, int n){
  for(int k = 0; k < n; k++)
    vec_b[k] = vec_a[k];
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
