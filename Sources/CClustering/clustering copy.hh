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
    double* similarities;
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
        int create_clusters(const double** embeddings, const int hidden_size, const int nb_pages, const double threshold, ClusteringResult* result);
};

template<typename T1>
std::ostream& operator <<( std::ostream& out, const std::vector<T1>& object )
{
    out << "[";
    if ( !object.empty() )
    {
        for(typename std::vector<T1>::const_iterator
            iter = object.begin();
            iter != --object.end();
            ++iter) {
                out << std::setprecision(4) << *iter << ", ";
        }
        out << *--object.end();
    }
    out << "]";
    return out;
}

template<typename T>
std::ostream & operator<<(std::ostream & os, std::set<T> vec)
{
    os<<"{";
    if(vec.size()!=0)
    {
        auto it = vec.end();
        if (vec.size() == 1) {
            it--;
        } else {
            it--;
            //it--;
        }
        
        std::copy(vec.begin(), it, std::ostream_iterator<T>(os, ", "));
        //os<<*--vec.end();
    }
    os<<"}";
    return os;
}

#endif /* clustering_hpp */
