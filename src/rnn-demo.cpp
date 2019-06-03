#include <stdio.h>
#include "rnn.h"
#include "tests.h"
#include "app.h"

void name() {
  // printf("labas\n", "");
}

int main (int c, char *v[]) {

  printf("RNN - demo\n\n");

  //run_tests();

  LanguageModel *model = new LanguageModel();
  model->doSomething();


  //run_cuda_sample();


 return 0;
}
