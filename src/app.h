#ifndef APP_HEADER
#define APP_HEADER

#include "rnn.h"

#include <cstring>
#include <vector>
#include <cstdlib>

class LanguageModel {
  private:
  public:
    LanguageModel();

    void doSomething();

  private:
    std::vector<DataNode*>* loadFromFile(const char *abc, const char *filename);

};

double* char_to_vec(const char* abc, char c);
char vec_to_char(const char* abc, double  *vec);

DataNode* str_to_nodes(const char* abc, const char* str);
void nodes_to_str(const char* abc, DataNode* node, char* str);




#endif /* APP_HEADER */
