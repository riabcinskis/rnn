#include "app.h"

LanguageModel::LanguageModel(){

}

void LanguageModel::doSomething(){

   //char abc[64]=" abcdefghijklmnopqrstuvwxyz-";
   char abc[64]=" 0123456789";

  //
  // int M = strlen(abc);
  // int I = strlen(abc);
  //
  // int V = 4;
  // Topology **topology = new Topology*[V];
  // for(int v = 0; v < V; v++){
  //   topology[v] = new Topology();
  //   topology[v]->addLayer(I);
  //   topology[v]->addLayer(M);
  // }
  //
  //
  // RnnCell *rnnCell = new RnnCell(M, topology);
  // Rnn* rnn = new Rnn(I, M, rnnCell);
  //
  // SecondMarkLimit* markLimit = new SecondMarkLimit(0, M);
  //
  // DataNode *inputNodes = str_to_nodes(abc, "labai prasau veik ");
  // DataNode* outputNodes = rnn->feedForward(inputNodes, markLimit);
  //
  // char str[512]="";
  // nodes_to_str(abc, outputNodes, str);
  //
  // printf("%s", str);

  std::vector<DataNode*>* nodeVector = loadFromFile(abc, "../files/data.txt");



  char str[512]="";
  for(int i = 0; i < nodeVector->size(); i++){
    nodes_to_str(abc, (*nodeVector)[i], str);
    printf("[%d] : %s\n", i, str);
  }



  double alpha = 0.8;
  double eta = 0.2;

  int M = strlen(abc);
  int V = 4;
  int I = strlen(abc);
  Topology **topology = new Topology*[V];
  for(int v = 0; v < V; v++){
    topology[v] = new Topology();
    topology[v]->addLayer(I);
    topology[v]->addLayer(M);
  }

  RnnCell *rnnCell = new RnnCell(M, topology);
  Rnn *rnn = new Rnn(I, M, rnnCell, RNN_FULL_BACKPROPAGATION);

  SecondMarkLimit* markLimit = new SecondMarkLimit(0, M);

  for(int n = 0; n < 100; n++){
    double iterError = 0;
    for(int i = 0; i < nodeVector->size(); i++){
      DataNode* input = (*nodeVector)[i];


      DataNode* startOutput = input;

      for(int i = 0; i < 3; i++)
        startOutput = startOutput->next;


      DataNode* output = NULL;
      int offset = 0;

      int partCount = 0;
      double sentenceError = 0.0;

      do{
        double partError;
        output = startOutput;
        for(int i = 0; i < offset; i++)
          output = output->next;
        if(rnn->backPropagation(input, output, markLimit, partError) == false){
          rnn->resetErrorDerivatives();
          break;
        }
        rnn->updateWeights(alpha, eta);
        rnn->resetErrorDerivatives();

        sentenceError += partError;
        partCount++;


        offset++;
      }while(true);
      sentenceError = sentenceError / (double)partCount;
      //printf("  error=%.4e\n", sentenceError);

      iterError += sentenceError;
    }

     printf("%.10f\n",iterError);


  }



  ////
  ////
  ////
  // DataNode *inputNodes = str_to_nodes(abc, "there is no sin");
  // DataNode* outputNodes = rnn->feedForward(inputNodes, markLimit);
  //
  // char str[512]="";
  // nodes_to_str(abc, outputNodes, str);
  //
  // printf(">>>>>>>>%s", str);




}

std::vector<DataNode*>* LanguageModel::loadFromFile(const char *abc, const char *filename){
  FILE* file = fopen("../files/data.txt", "r");

  char* line = NULL;
  size_t len = 0;
  ssize_t read;

  std::vector<DataNode*>* vector = new std::vector<DataNode*>();

  while ((read = getline(&line, &len, file)) != -1) {
    printf("\n\n");
    printf("%s", line);

    DataNode* node = str_to_nodes(abc, line);
    vector->push_back(node);


  }

  fclose(file);
  return vector;
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
