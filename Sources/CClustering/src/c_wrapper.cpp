//
//  c_wrapper.cpp
//
//  Created by Julien Plu on 09/05/2022.
//

#include "../clustering.hpp"


extern "C" void* createClustering(const float threshold, const char* model_path, uint16_t hidden_size, const char* tokenizer_model_path, uint16_t max_seq_length) {
    Clustering* clustering = new Clustering(threshold, model_path, hidden_size, tokenizer_model_path, max_seq_length);
    
    return (void*) clustering;
}

extern "C" int add_textual_item(void* handle, const char* text, const int idx, struct ClusteringResult* result) {
    Clustering* clustering = (Clustering*)handle;
    
    return clustering->add_textual_item(text, idx, result);
}

extern "C" int remove_textual_item(void* handle, const int idx, const int from_add, struct ClusteringResult* result) {
    Clustering* clustering = (Clustering*)handle;
    
    return clustering->remove_textual_item(idx, from_add, result);
}

extern "C" int recompute_clustering_threshold(void* handle, const struct ClusterDefinition* expected_clusters, struct ClusteringResult* result) {
    Clustering* clustering = (Clustering*)handle;
    
    return clustering->recompute_clustering_threshold(expected_clusters, result);
}

extern "C" float get_threshold(void* handle) {
    Clustering* clustering = (Clustering*)handle;
    
    return clustering->get_threshold();
}

extern "C" void removeClustering(void* handle) {
    Clustering* clustering = (Clustering*)handle;
    
    delete clustering;
}
