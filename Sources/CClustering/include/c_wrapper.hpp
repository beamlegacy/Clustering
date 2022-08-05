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
    uint16_t size;
    float performance;
};

struct TokenizerResult {
    uint32_t* input_ids;
    uint16_t size;
    float performance;
};

struct ClusteringResult {
    uint16_t* indices;
    uint16_t indices_size;
    uint16_t* clusters_split;
    uint16_t clusters_split_size;
    float* similarities;
    float performance;
};


void* createModel(const char* model_path, uint16_t hidden_size);
int predict(void* handle, const struct TokenizerResult* tokenizer_result, struct ModelResult* result);
void removeModel(void* handle);

void* createTokenizer(const char* tokenizer_path, uint16_t max_seq_length);
int tokenize(void* handle, const char* text, struct TokenizerResult* result);
void removeTokenizer(void* handle);

void* createClustering();
int create_clusters(void* handle, const float** embeddings, const uint16_t hidden_size, const uint16_t nb_pages, const float threshold, struct ClusteringResult* result);
void removeClustering(void* handle);

#endif /* c_wrapper_hpp */
