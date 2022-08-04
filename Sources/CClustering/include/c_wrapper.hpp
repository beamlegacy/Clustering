//
//  c_wrapper.hpp
//
//  Created by Julien Plu on 07/07/2022.
//

#ifndef c_wrapper_hpp
#define c_wrapper_hpp

#include <stdint.h>

struct ModelResult {
    float* weigths;
    unsigned long size;
    float performance;
};

struct TokenizerResult {
    int32_t* input_ids;
    unsigned long size;
    float performance;
};

struct ClusteringResult {
    int* indices;
    unsigned long indices_size;
    unsigned long* clusters_split;
    unsigned long clusters_split_size;
    double* similarities;
    float performance;
};


void* createModel(const char* model_path, int32_t hidden_size);
int predict(void* handle, const struct TokenizerResult* tokenizer_result, struct ModelResult* result);
void removeModel(void* handle);

void* createTokenizer(const char* tokenizer_path, int32_t max_seq_length);
int tokenize(void* handle, const char* text, struct TokenizerResult* result);
void removeTokenizer(void* handle);

void* createClustering();
int create_clusters(void* handle, const double** embeddings, const int hidden_size, const int nb_pages, struct ClusteringResult* result);
void removeClustering(void* handle);

#endif /* c_wrapper_hpp */
