#include "app.h"

LanguageModel::LanguageModel(){

}

void LanguageModel::doSomething(){

  char abc[64]=" abcdefghijklmnopqrstuvwxyz";

  int M = strlen(abc);
  int I = strlen(abc);

  int V = 4;
  Topology **topology = new Topology*[V];
  for(int v = 0; v < V; v++){
    topology[v] = new Topology();
    topology[v]->addLayer(I);
    topology[v]->addLayer(M);
  }


  RnnCell *rnnCell = new RnnCell(M, topology);
  Rnn* rnn = new Rnn(I, M, rnnCell);

  SecondMarkLimit* markLimit = new SecondMarkLimit(0, M);

  DataNode *inputNodes = str_to_nodes(abc, "labai prasau veik ");
  DataNode* outputNodes = rnn->feedForward(inputNodes, markLimit);

  char str[512]="";
  nodes_to_str(abc, outputNodes, str);

  printf("%s", str);

}

//
// GLOBAL
//
void char_to_vec(const char* abc, double* vec, char c){
  int n = strlen(abc);


  for(int i = 0; i < n; i++)
    vec[i] = abc[i] == c ? 1.0 : 0.0;

}

char vec_to_char(const char* abc, double *vec){
  int n = strlen(abc);
  int maxAt = 0;
  for(int i = 1; i < n; i++)
    maxAt = vec[i] > vec[maxAt] ? i : maxAt;

  return abc[maxAt];
}

DataNode* str_to_nodes(const char* abc, const char* str){
  int n = strlen(abc);

  DataNode* out = new DataNode(n);
  DataNode* q = out;
  int index = 0;
  while(str[index] != '\0'){
    char_to_vec(abc, q->vec, str[index]);
    index++;

    if(str[index] == '\0') break;
    q->next = new DataNode(n);
    q = q->next;
  }

  return out;

}

void nodes_to_str(const char* abc, DataNode* node, char* str){
  int n = strlen(abc);
  DataNode* q = node;

  int index = 0;
  while(q != NULL){
    str[index++] = vec_to_char(abc, q->vec);
    q = q->next;
  }

  str[index] = '\0';

}
