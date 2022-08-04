//
//  clustering.hpp
//
//  Created by Julien Plu on 07/07/2022.
//

#ifndef clustering_hpp
#define clustering_hpp

#include "sentencepiece_processor.hpp"
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>
#include <set>



#include <iomanip>


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
    float performance;
};

class Tokenizer {
    private:
        sentencepiece::SentencePieceProcessor tokenizer;
        int32_t max_seq_length;
    public:
        Tokenizer(const char* tokenizer_model_path, int32_t max_seq_length);
        int tokenize(const char* text, TokenizerResult* result);
};

class Model {
    private:
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::Env> env;
        int32_t hidden_size;
    public:
        Model(const char* model_path, int32_t hidden_size);
        int predict(const TokenizerResult* tokenizer_result, ModelResult* result);
};

class Clustering {
    private:
        //float threshold = 0.3105;
        //std::vector<std::vector<float>> embeddings;
        //std::vector<std::string> uuids;
        std::vector<std::vector<double>> similarities;
        double norm(const std::vector<double> &vector);
        std::vector<double> normalize(const std::vector<double> &vector);
        double cosine_similarity(const std::vector<double> &vector1, const std::vector<double> &vector2);
        void cosine_similarity_matrix(const std::vector<std::vector<double>> &embeddings);
        std::vector<int> argsort(const std::vector<double> &array);
        std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<int>>> topk_matrix(const int k);
        std::tuple<std::vector<double>, std::vector<int>> topk(const int k, const std::vector<double> &array);
    public:
        Clustering();
        //void addTextualItem(const char* uuid, const float* embedding, const int hidden_size);
        const double** getSimilarities();
        //void removeTextualItem(const char* uuid);
        int create_clusters(const double** embeddings, const int hidden_size, const int nb_pages, ClusteringResult* result);
};

#endif /* clustering_hpp */
