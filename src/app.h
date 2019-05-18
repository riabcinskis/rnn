#ifndef APP_HEADER
#define APP_HEADER

#include "rnn.h"

#include <cstring>

class LanguageModel {
  private:
  public:
    LanguageModel();

    void doSomething();
};

double* char_to_vec(const char* abc, char c);
char vec_to_char(const char* abc, double  *vec);

DataNode* str_to_nodes(const char* abc, const char* str);
void nodes_to_str(const char* abc, DataNode* node, char* str);




#endif /* APP_HEADER */
