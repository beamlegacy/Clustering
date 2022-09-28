//
//  c_wrapper.hpp
//
//  Created by Julien Plu on 07/07/2022.
//

#ifndef c_wrapper_hpp
#define c_wrapper_hpp

#include <stdint.h>


struct ClusterDefinition {
    uint16_t* indices;
    uint16_t indices_size;
    uint16_t* clusters_split;
    uint16_t clusters_split_size;
};

struct ClusteringResult {
    struct ClusterDefinition* cluster;
    float performance;
};


void* createClustering(const float threshold, const char* model_path, uint16_t hidden_size, const char* tokenizer_model_path, uint16_t max_seq_length);
int add_textual_item(void* handle, const char* text, const int idx, struct ClusteringResult* result);
int remove_textual_item(void* handle, const int idx, const int from_add, struct ClusteringResult* result);
int recompute_clustering_threshold(void* handle, const struct ClusterDefinition* expected_clusters, struct ClusteringResult* result);
float get_threshold(void* handle);
void removeClustering(void* handle);

#endif /* c_wrapper_hpp */
