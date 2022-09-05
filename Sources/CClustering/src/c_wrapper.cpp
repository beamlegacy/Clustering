//
//  c_wrapper.cpp
//
//  Created by Julien Plu on 09/05/2022.
//

#include "../clustering.hpp"


extern "C" void* createModel(const char* model_path, uint16_t hidden_size) {
    Model* model = new Model(model_path, hidden_size);
    
    return (void*) model;
}

extern "C" int predict(void* handle, const struct TokenizerResult* tokenizer_result, struct ModelResult* result) {
    Model* model = (Model*)handle;
    
    return model->predict(tokenizer_result, result);
}

extern "C" void removeModel(void* handle) {
    Model* model = (Model*)handle;
    
    delete model;
}

extern "C" void* createTokenizer(const char* tokenizer_path, uint16_t max_seq_length) {
    Tokenizer* tokenizer = new Tokenizer(tokenizer_path, max_seq_length);
    
    return (void*) tokenizer;
}

extern "C" int tokenize(void* handle, const char* text, struct TokenizerResult* result) {
    Tokenizer* tokenizer = (Tokenizer*)handle;
    
    return tokenizer->tokenize(text, result);
}

extern "C" void removeTokenizer(void* handle) {
    Tokenizer* tokenizer = (Tokenizer*)handle;
    
    delete tokenizer;
}

extern "C" void* createClustering() {
    Clustering* clustering = new Clustering();
    
    return (void*) clustering;
}

extern "C" int create_clusters(void* handle, const float** embeddings, const uint16_t hidden_size, const uint16_t nb_pages, struct ClusteringResult* result) {
    Clustering* clustering = (Clustering*)handle;
    
    return clustering->create_clusters(embeddings, hidden_size, nb_pages, result);
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
